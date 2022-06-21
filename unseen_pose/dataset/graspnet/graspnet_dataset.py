import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import sys
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
import MinkowskiEngine as ME
import scipy.io as scio
import open3d as o3d
import copy
import cv2
import json
import logging

from transforms3d.axangles import axangle2mat

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

from graspnetAPI.utils.utils import create_point_cloud_from_depth_image, CameraInfo, parse_posevector
from graspnetAPI.utils.xmlhandler import xmlReader
from unseen_pose.utils import get_posed_points, torch_get_closest_point, convert_data_to_device
from unseen_pose.constant import GRASPNET_ROOT, GRASPNET_ROOT_REAL, RANDOM_SEED, MODEL_DOWNSAMPLED_DIR, LABEL_DIR, \
    VOXEL_SIZE, MODEL_DOWNSAMPLED_DIR_REAL
from unseen_pose.dataset.graspnet.graspnet_constant import PAIR_NUM, SCENE_PAIR_NUM, ANN_NUM, RESIZE_WIDTH, \
    RESIZE_HEIGHT, ORIGIN_HEIGHT, ORIGIN_WIDTH, CLOSE_DIST_THRES, REMOTE_DIST_THRES


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def get_matches(ret_dict, num_sample, labels):
    '''
    labels: (num_points, 2)
    return: (num_sample, 2), type=torch.int32
    '''
    # labels_row = labels.reshape((-1, 3))  # (HEIGHT* WIDTH,3)
    idxs_l = ret_dict['l']['idxs']  # [N_l]
    idxs_r = ret_dict['r']['idxs']

    # print(idxs_l.shape, labels.shape)
    logger.debug(f"labels shape:{labels.shape}")
    if ret_dict['l']['mask'] is not None:
        logger.debug(f"mask is not none:{ret_dict['l']['mask'].shape}")
        obj_mask_l = ret_dict['l']['mask'].reshape(
            (-1, 1)).squeeze()[idxs_l]  # (N_l,)
        logger.debug(f"obj_mask_l:{obj_mask_l}")
        logger.debug(f'new_mask:{obj_mask_l.shape}')
    else:
        obj_mask_l = torch.ones(len(labels), dtype=bool)  # (N_l,)
        logger.debug(f'mask is None, new mask:{obj_mask_l.shape}')
    # obj_mask_r = ret_dict['r']['mask'].reshape((-1, 1)).squeeze()

    label_mask_l = labels[:, 1] < CLOSE_DIST_THRES  # (N_l,)
    logger.debug(f"label_mask_l:{label_mask_l}")
    mask_l = obj_mask_l & label_mask_l  # (N_l,)
    if torch.sum(mask_l) < 1.0:
        # logging.warning(f"No match find")
        return None
    logger.debug(f"mask_l:{mask_l.shape}")
    mask_l_indices = torch.nonzero(mask_l)
    # print(f'mask_l_indices:{mask_l_indices.shape}')

    mask_l_indices = mask_l_indices[:, 0]
    # print(f'mask_l_indices after:{mask_l_indices.shape}')
    logging.debug(f'mask_l_indices:{mask_l_indices.shape}')

    # mask_r of idxs
    # idxs_mask_r = torch.zeros((RESIZE_HEIGHT * RESIZE_WIDTH)).int() # TODO: I am not sure whether right shape is always (H,W,3)
    # idxs_mask_r[idxs_r] = 1
    # mask_r = torch.ones(len(labels), dtype = bool) # (N_l,)
    # mask_r_indices = torch.nonzero(mask_r)
    # print('dim', mask_l_indices.dim())
    if mask_l_indices.dim() == 0:
        print(f'mask_l_indices:{mask_l_indices},type:{type(mask_l_indices)}')
    match_id = torch.randint(len(mask_l_indices), (num_sample,), dtype=torch.int64)  # (num_sample, )
    logging.debug(f'match_id:{match_id}')
    matching = torch.empty((num_sample, 2), dtype=int)  # (num_sample, 2)
    matching[:, 0] = mask_l_indices[match_id].long()  # (num_sample, )
    logging.debug(f'matching[:. 0]:{matching[:, 0]}')
    matching[:, 1] = labels[matching[:, 0], 0].long()  # (num_sample, )

    return matching


def get_non_matches(ret_dict, num_sample, if_mask_r=True):
    '''
    return: (num_sample, 2)
    '''
    # labels_row = labels.reshape((-1, 3))  # (HEIGHT* WIDTH,3)
    idxs_l = ret_dict['l']['idxs']  # [N_l]
    idxs_r = ret_dict['r']['idxs']
    N_l = len(idxs_l)
    N_r = len(idxs_r)
    if ret_dict['l']['mask'] is not None:
        # logger.debug(f"mask is not none:{ret_dict['l']['mask'].shape}")
        obj_mask_l = ret_dict['l']['mask'].reshape(
            (-1, 1)).squeeze()[idxs_l]  # (N_r,)
    else:
        obj_mask_l = torch.ones(N_l, dtype=bool)  # (N_l,)
    if ret_dict['r']['mask'] is not None:
        # logger.debug(f"mask is not none:{ret_dict['l']['mask'].shape}")
        obj_mask_r = ret_dict['r']['mask'].reshape(
            (-1, 1)).squeeze()[idxs_r]  # (N_r,)
    else:
        obj_mask_r = torch.ones(N_r, dtype=bool)  # (N_l,)

    mask_l_indices = torch.nonzero(obj_mask_l)
    mask_l_indices = mask_l_indices[:, 0]
    # logging.debug(f'mask_l_indices:{mask_l_indices.shape}')
    match_id_l = torch.randint(len(mask_l_indices), (num_sample,), dtype=torch.int64)  # (num_sample, )

    mask_r_indices = torch.nonzero(obj_mask_r)
    mask_r_indices = mask_r_indices[:, 0]
    # logging.debug(f'mask_l_indices:{mask_l_indices.shape}')
    match_id_r = torch.randint(len(mask_r_indices), (num_sample,), dtype=torch.int64)  # (num_sample, )

    non_match = torch.empty((num_sample, 2), dtype=torch.int64)
    non_match[:, 0] = mask_l_indices[match_id_l].long()  # (num_sample, )
    non_match[:, 1] = mask_r_indices[match_id_r].long()  # (num_sample, )

    return non_match


class GraspNetDataset(Dataset):
    def __init__(self, root, camera='realsense', split='all', voxel_size=VOXEL_SIZE, n_match=10, n_non_match=10,
                 model_pair_ratio=1.0, scene_pair_ratio=1.0, collect_direction=1, list_file_dir='graspnet_offline_list', label_dir=None,
                 matching_devices='cpu', generate_label=False, use_remove_mask=True, use_augmentation=True,
                 mask_switch_frequency=3, add_gaussian_noise_frequency=3, save_intermediate_product_frequency=0,
                 inference=False):
        self.root = root
        self.camera = camera
        self.split = split
        self.voxel_size = voxel_size
        self.n_match = n_match
        self.n_non_match = n_non_match
        self.label_dir = label_dir
        self.inference = inference  # if the dataset is used for inference or not
        self.collect_direction = collect_direction
        self.matching_device = matching_devices
        self.generate_label = generate_label
        self.model_pair_ratio = model_pair_ratio
        self.scene_pair_ratio = scene_pair_ratio
        self.use_remove_mask = use_remove_mask
        self.use_augmentation = use_augmentation
        self.mask_switch_frequency = mask_switch_frequency
        self.add_gaussian_noise_frequency = add_gaussian_noise_frequency
        self.save_intermediate_product_frequency = save_intermediate_product_frequency

        print(self.matching_device, torch.cuda.device_count())

        if self.camera != 'blender_proc':
            if split == 'all':
                self.sceneIds = list(range(190))
            elif split == 'train':
                self.sceneIds = list(range(100))
            elif split == 'test':
                self.sceneIds = list(range(100, 190))
            elif split == 'test_seen':
                self.sceneIds = list(range(100, 130))
            elif split == 'test_similar':
                self.sceneIds = list(range(130, 160))
            elif split == 'test_novel':
                self.sceneIds = list(range(160, 190))
        else:
            assert split in ['all', 'train', 'test_seen', 'test']
            if split in ['all', 'train']:
                self.sceneIds = list(range(1400))
            elif split == 'test_seen':
                self.sceneIds = list(range(1400, 1500))
            elif split == 'test':
                self.sceneIds = list(range(1400, 1500))

        if list_file_dir is not None:
            print(f'\033[034mLoading Scene Pair List file:{os.path.join(list_file_dir, split + ".npy")}\033[0m')
            self.scene_pair_list = np.load(os.path.join(list_file_dir, split + '.npy'))  # (total sceen pair num, 3)
            if self.use_remove_mask:
                print(
                    f'\033[034mLoading Scene Pair List Remove Mask file:{os.path.join(list_file_dir, split + "_remove_mask.npy")}\033[0m')
                self.scene_pair_remove_mask = np.load(os.path.join(list_file_dir, split + '_remove_mask.npy'))
                self.scene_pair_list = self.scene_pair_list[~self.scene_pair_remove_mask]

            total_scene_pair_num = len(self.scene_pair_list)
            selected_total_scene_pair_num = int(total_scene_pair_num * self.scene_pair_ratio)
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(self.scene_pair_list)
            if self.collect_direction > 0:
                self.scene_pair_list = self.scene_pair_list[:selected_total_scene_pair_num, :3]
            else:
                if selected_total_scene_pair_num == 0:
                    self.scene_pair_list = self.scene_pair_list[:0, :3]
                else:
                    self.scene_pair_list = self.scene_pair_list[-selected_total_scene_pair_num:, :3]

            print(
                f'\033[034mTotal Scene Pair Number:{total_scene_pair_num}, Selected:{selected_total_scene_pair_num}\033[0m')

            print(
                f'\033[034mLoading Model Pair List file:{os.path.join(list_file_dir, "model_" + split + ".npy")}\033[0m')
            self.model_pair_list = np.load(
                os.path.join(list_file_dir, "model_" + split + ".npy"))  # (total model pair num, 3)
            if self.use_remove_mask:
                print(
                    f'\033[034mLoading Model Pair List Remove Mask file:{os.path.join(list_file_dir, "model_" + split + "_remove_mask.npy")}\033[0m')
                self.model_pair_remove_mask = np.load(
                    os.path.join(list_file_dir, "model_" + split + '_remove_mask.npy'))
                self.model_pair_list = self.model_pair_list[~self.model_pair_remove_mask]

            total_model_pair_num = len(self.model_pair_list)

            selected_total_model_pair_num = int(total_model_pair_num * self.model_pair_ratio)
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(self.model_pair_list)
            if self.collect_direction > 0:
                self.model_pair_list = self.model_pair_list[:selected_total_model_pair_num]
            else:
                if selected_total_model_pair_num > 0:
                    self.model_pair_list = self.model_pair_list[-selected_total_model_pair_num:]
                else:
                    self.model_pair_list = self.model_pair_list[:0]
            print(
                f'\033[034mTotal Model Pair Number:{total_model_pair_num}, Selected:{selected_total_model_pair_num}\033[0m')

            # model_pair_num = int(len(self.pair_list) / (1 - self.model_pair_ratio) * self.model_pair_ratio)
            # if model_pair_num > len(self.model_pair_list):
            #     raise ValueError('Not Enough model pair')
            self.model_pair_mask = np.concatenate(
                (np.zeros(len(self.scene_pair_list)), np.ones(len(self.model_pair_list)))).astype(
                bool)  # (pair num + model_pair_num)
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(self.model_pair_mask)
            self.all_pair_list = np.empty((len(self.model_pair_list) + len(self.scene_pair_list), 3),
                                          dtype=int)  # (pair num + model_pair_num, 3)
            self.all_pair_list[self.model_pair_mask] = self.model_pair_list  # (model_pair_num, 3)
            self.all_pair_list[~self.model_pair_mask] = self.scene_pair_list  # (pair num, 3)
            # print(f'all_pair_list:{self.all_pair_list.shape}:\n{self.all_pair_list}')
            self.from_list = True
        else:
            self.from_list = False
            # raise ValueError('Not from list is not supported now')
        self.sceneIds = ['scene_{}'.format(
            str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.metapath = []
        self.maskpath = []
        self.scenename = []
        self.frameid = []
        self.ann_posespath = []

        resize = transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH))
        totensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans_list = [resize, totensor]
        self.rgb_transform = transforms.Compose(trans_list)
        self.camera_poses = np.load(os.path.join(
            root, 'scenes', 'scene_0000', camera, 'camera_poses.npy'))

        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            if self.camera != 'blender_proc':
                self.ann_posespath.append(os.path.join(
                    root, 'scenes', x, 'rs_wrt_kn.npy'))
            for img_num in range(ANN_NUM):
                self.colorpath.append(os.path.join(
                    root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(
                    root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                if self.camera != 'blender_proc':
                    self.metapath.append(os.path.join(
                        root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                    self.maskpath.append(os.path.join(
                        root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

    def __len__(self):
        if self.from_list:
            return len(self.all_pair_list)
        else:
            return len(self.sceneIds) * ANN_NUM * (ANN_NUM - 1)

    def load_rgb_depth(self, scene_id, ann_id):
        '''
        return:
        rgb: torch.tensor [3, RESIZE_HEIGHT, RESIZE_WIDTH]
        cloud: torch.tensor [3, RESIZE_HEIGHT, RESIZE_WIDTH]
        '''
        # index = scene_id * ANN_NUM + ann_id
        color_path = os.path.join(self.root, 'scenes', 'scene_%04d' %
                                  scene_id, self.camera, 'rgb', '%04d.png' % ann_id)
        rgb = Image.open(color_path)
        rgb = self.rgb_transform(rgb).numpy()
        depth_path = os.path.join(self.root, 'scenes', 'scene_%04d' %
                                  scene_id, self.camera, 'depth', '%04d.png' % ann_id)
        depth = Image.open(depth_path)
        depth = np.array(depth.resize(
            (RESIZE_WIDTH, RESIZE_HEIGHT), Image.NEAREST))
        return rgb, depth

    def augment_data(self, point_cloud, is_object):
        # Flipping along the YZ plane
        # if np.random.random() > 0.5:
        #     flip_mat = np.array([[-1, 0, 0],
        #                          [ 0, 1, 0],
        #                          [ 0, 0, 1]])
        #     point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
        #     for i in range(len(object_poses_list)):
        #         object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        if not is_object:
            rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4  # -45 ~ +45 degree
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            rot_mat = np.array([[c, -s, 0],
                                [s, c, 0],
                                [0, 0, 1]])
        else:
            axe = np.random.rand(3)
            axe = axe / np.linalg.norm(axe)
            angle = 2 * np.random.random() * np.pi
            rot_mat = axangle2mat(axe, angle)
        rot_mat = rot_mat.astype(np.float32)
        rot_mat = torch.from_numpy(rot_mat).to(point_cloud.device)

        point_cloud = torch.matmul(point_cloud, rot_mat)
        return point_cloud, rot_mat

    def add_gaussian_noise(self, point_cloud, is_object, density=0.0005):
        if not is_object:
            point_cloud = point_cloud + density * torch.randn_like(point_cloud)
        else:
            noise = density * torch.randn_like(point_cloud)
            noise[:, :2] = 0.0
            point_cloud = point_cloud + noise
        return point_cloud

    def load_mask(self, scene_id, ann_id):
        '''
        return:
        mask: torch.uint8 of the mask.
        '''
        # index = scene_id * ANN_NUM + ann_id
        mask_path = os.path.join(self.root, 'scenes', 'scene_%04d' %
                                 scene_id, self.camera, 'label', '%04d.png' % ann_id)
        # print(mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        from collections import Counter
        if self.camera not in ['blender_proc', 'realsense', 'kinect']:
            mask = cv2.resize(mask[:, :, 0], (384, 288))
        else:
            mask = cv2.resize(mask[:, :], (384, 288),
                              interpolation=cv2.INTER_NEAREST)
        # print(np.unique(mask))
        mask = mask.astype(np.int32)
        return torch.tensor(mask)

    def load_rgb_with_pointcloud(self, scene_id, ann_or_obj_id, is_object):
        '''
        return:
        rgb: torch.tensor [3, RESIZE_HEIGHT, RESIZE_WIDTH] for scene, 
            (n, 3) for object.
        cloud: torch.tensor [3, RESIZE_HEIGHT, RESIZE_WIDTH] for scene,
            (n, 3) for object.
        '''
        if is_object:
            if self.camera == 'blender_proc':
                pcd = o3d.io.read_point_cloud(os.path.join(MODEL_DOWNSAMPLED_DIR, '%03d.ply' % ann_or_obj_id))
            else:
                pcd = o3d.io.read_point_cloud(os.path.join(MODEL_DOWNSAMPLED_DIR_REAL, '%03d.ply' % ann_or_obj_id))
            xyz = np.asarray(pcd.points)
            if self.camera in ['blender_proc']:
                xyz = 0.001 * xyz
            pcd.points = o3d.utility.Vector3dVector(xyz)
            # print(os.path.join(MODEL_DOWNSAMPLED_DIR, '%03d.ply' % ann_or_obj_id))
            rgb = torch.from_numpy(np.asarray(pcd.colors))  # (n, 3)
            cloud = torch.from_numpy(np.asarray(pcd.points))  # (n, 3)
        else:
            color_path = os.path.join(self.root, 'scenes', 'scene_%04d' %
                                      scene_id, self.camera, 'rgb', '%04d.png' % ann_or_obj_id)
            depth_path = os.path.join(self.root, 'scenes', 'scene_%04d' %
                                      scene_id, self.camera, 'depth', '%04d.png' % ann_or_obj_id)
            # index = scene_id * ANN_NUM + ann_id
            rgb = Image.open(color_path)
            rgb = self.rgb_transform(rgb)
            depth = Image.open(depth_path)
            depth = np.array(depth.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.NEAREST))
            # if self.camera != "blender_proc":
            meta_path = os.path.join(self.root, 'scenes', 'scene_%04d' %
                                     scene_id, self.camera, 'meta', '%04d.mat' % ann_or_obj_id)
            meta = scio.loadmat(meta_path)
            try:
                intrinsic = meta['intrinsic_matrix']
                # intrinsic = np.array([2133.55615234375, 0.0, 626.4738159179688, 0.0, 2134.97412109375, 363.1217956542969, 0.0, 0.0, 1.0]).reshape(3,3)
                factor_depth = meta['factor_depth']
            except Exception as e:
                print(repr(e))
                print(scene_id)
            # else:
            #     intrinsic = [[2133.55615234375, 0.0, 626.4738159179688],
            #                  [0.0, 2134.97412109375, 363.1217956542969],
            #                  [0.0, 0.0, 1.0]]
            #     factor_depth = 1000

            sx = RESIZE_WIDTH / ORIGIN_WIDTH
            sy = RESIZE_HEIGHT / ORIGIN_HEIGHT
            camera = CameraInfo(RESIZE_WIDTH, RESIZE_HEIGHT, sx * intrinsic[0][0], sy * intrinsic[1][1],
                                sx * intrinsic[0][2] + 0.5 * sx - 0.5, sy * intrinsic[1][2] + 0.5 * sy - 0.5,
                                factor_depth)

            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
            cloud = torch.tensor(np.transpose(cloud, (2, 0, 1))).float()

        return rgb, cloud

    def match_points(self, point_l, point_r, l2r_pose, use_icp=True, computing_device='cpu'):
        '''
        :param point_l: numpy (N_l,3)
        :param point_r: numpy array (N_r,3)
        :param l2r_pose: transformation matrix of the points that from left to right.
        :return:(n_l, 2)
        '''
        import time
        t11 = time.time()
        pcd_r = o3d.geometry.PointCloud()
        pcd_r.points = o3d.utility.Vector3dVector(point_r)
        pcd_r.transform(np.linalg.inv(l2r_pose))

        t12 = time.time()
        pcd_l = o3d.geometry.PointCloud()
        pcd_l.points = o3d.utility.Vector3dVector(point_l)

        if use_icp:
            threshold = 0.02
            reg_p2p = o3d.pipelines.registration.registration_icp(pcd_r, pcd_l, threshold, np.eye(
                4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
            pcd_r.transform(reg_p2p.transformation)

        trans_points_r = torch.from_numpy(np.asarray(pcd_r.points)).float()
        dist, min_idx = torch_get_closest_point(point_l, trans_points_r, computing_device=computing_device)
        label = np.empty((len(point_l), 2), dtype=float)
        label[:, 0] = min_idx.cpu().numpy()
        label[:, 1] = dist.cpu().numpy()
        t13 = time.time()
        return label

    def get_ids(self, index):
        scene_id = index // SCENE_PAIR_NUM
        pair_id = index % SCENE_PAIR_NUM
        ann_id_l = pair_id // (ANN_NUM - 1)
        ann_id_r = pair_id % (ANN_NUM - 1)
        if ann_id_r >= ann_id_l:
            ann_id_r += 1
        return scene_id, ann_id_l, ann_id_r

    def get_obj_6dpose(self, scene_id, ann_id, obj_id, camera):
        if self.camera != 'blender_proc':
            scene_reader = xmlReader(
                os.path.join(self.root, 'scenes', 'scene_%04d' % scene_id, camera, 'annotations', '%04d.xml' % ann_id))
            posevectors = scene_reader.getposevectorlist()
            for posevector in posevectors:
                obj_idx, pose = parse_posevector(posevector)
                if obj_idx == obj_id:
                    return pose
            return None
        else:
            with open(os.path.join(self.root, 'scenes', 'scene_%04d' % scene_id, camera, 'scene_gt.json')) as fp:
                scene_gt_dict = json.load(fp)
            ann = scene_gt_dict[str(ann_id)]
            for gt in ann:
                if gt['obj_id'] == obj_id:
                    R = np.array(gt['cam_R_m2c']).reshape(3, 3)
                    t = 0.001 * np.array(gt['cam_t_m2c']).reshape(1, 3)
                    pose = np.zeros((4, 4))
                    pose[:3, :3] = R
                    pose[:3, 3] = t
                    pose[3, 3] = 1.000
                    return pose
            return None

    def __getitem__(self, *args):
        '''
        :param args: support two types get item, index or
         scene_num, ann_id_l, ann_id_r, is_object
        :return:
        a dict of info
        '''
        import time
        t1 = time.time()
        if len(args) == 1:
            index = args[0]
            if self.from_list:
                scene_id, l_id, ann_id_r = self.all_pair_list[index]
                is_object = self.model_pair_mask[index]
            else:
                scene_id, l_id, ann_id_r = self.get_ids(index)
                raise NotImplementedError('not from list not implemented')

        elif len(args) == 4:
            # (0,2,122,True) -> scene:0, object or left ann id: 2, right ann id: 122, left point cloud is object.
            scene_id = args[0]
            l_id = args[1]
            ann_id_r = args[2]
            is_object = args[3]

        # scene_id, l_id, ann_id_r, is_object = 68, 850, 49, True
        # print(
        #     f'calling index: camera:{self.camera} scene_id:{scene_id},l_id:{l_id},ann_id_r:{ann_id_r},is_object:{is_object}')
        if is_object:
            label_path = os.path.join(self.label_dir, '%s-%03d-%03d-%04d-object.npy' %
                                      (self.camera, scene_id, l_id, ann_id_r))
        else:
            label_path = os.path.join(self.label_dir, '%s-%03d-%04d-%04d.npy' %
                                      (self.camera, scene_id, l_id, ann_id_r))
        # print(label_path)
        if self.generate_label:
            if os.path.exists(label_path):
                print(f'Using existed Label {label_path}')
                return 0
        ret_dict = {}
        # print(f'calling with scene:{scene_id}, annl:{ann_id_l}, annr:{ann_id_r}')

        rotation_matrix_for_augmentation = {
            'l': torch.eye(3),
            'r': torch.eye(3)
        }

        for lr_idx, ann_or_object_id in enumerate([l_id, ann_id_r]):
            # lr: 'l' for left, 'r' for right
            if lr_idx == 0:
                lr = 'l'
            elif lr_idx == 1:
                lr = 'r'
            else:
                raise ValueError('lr must be l or r')

            rgb, cloud = self.load_rgb_with_pointcloud(scene_id, ann_or_object_id, (is_object and (lr == 'l')))
            # print(lr, rgb.shape, cloud.shape)

            if is_object and (lr == 'l'):
                origin_cloud = cloud
                mask = None
                objectness_mask = torch.ones((cloud.size()[0],), dtype=torch.int32)
                sem_seg_mask = torch.ones((cloud.size()[0],), dtype=torch.int32)
            else:
                cloud = cloud.reshape((3, -1)).T
                origin_cloud = cloud.detach().clone()

                mask = self.load_mask(scene_id, ann_or_object_id)
                objectness_mask = (mask > 0.5)
                objectness_mask = objectness_mask.resize(cloud.size()[0])
                sem_seg_mask = (
                            mask == l_id + 1)  # ATTENTION DON'T FORGET PLUS ONE: 0 -> BACKGROUND, ID + 1 -> OBJECT ID
                sem_seg_mask = sem_seg_mask.resize(cloud.size()[0])

            coords = torch.tensor(np.ascontiguousarray(cloud / self.voxel_size, dtype=np.int32))
            _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
            coords = coords[idxs]
            cloud = cloud[idxs].float()  # (n, 3)
            if lr == 'r':
                objectness_label = objectness_mask[idxs]
                sem_seg_label = sem_seg_mask[idxs]
            else:
                objectness_label = objectness_mask[idxs]
                sem_seg_label = sem_seg_mask[idxs]

            # print(sem_seg_label.size())
            # print(f"scene_id:{scene_id},l_id:{l_id},ann_id_r:{ann_id_r},is_object:{is_object}, sem seg: ", sem_seg_label.sum())
            if sem_seg_label is not None and sem_seg_label.sum() <= 0:
                f = open(f'err_log.txt', 'a')
                print(
                    f'\033[31mMatches is None for {scene_id} {l_id} {ann_id_r},{is_object}\033[0m')
                f.write(
                    f'match_is_none,{self.camera},{scene_id},{l_id},{ann_id_r},{is_object}\n')
                f.close()
            if is_object and (lr == 'l'):
                rgb_selected = rgb[idxs]  # (n, 3)
            else:
                rgb_selected = rgb.permute(1, 2, 0).reshape((-1, 3))[idxs]  # (n, 3)

            # mask_ = sem_seg_mask if (index % self.mask_switch_frequency) != 0 else objectness_mask
            mask_ = objectness_mask
            ret_dict[lr] = {'rgb': rgb, 'idxs': idxs, 'coords': coords, 'rgb_selected': rgb_selected,
                            'cloud': cloud, 'origin_cloud': origin_cloud, 'mask': mask_,
                            'objectness_label': objectness_label, 'sem_seg_label': sem_seg_label}
        t2 = time.time()
        poses = self.camera_poses
        pose_r = poses[ann_id_r]

        if is_object:
            obj_pose = self.get_obj_6dpose(scene_id, ann_id_r, l_id, self.camera)
            l2r_pose = obj_pose
            use_icp = False
        else:
            pose_l = poses[l_id]
            l2r_pose = np.matmul(np.linalg.inv(pose_r), pose_l)
            use_icp = False

        if os.path.exists(label_path) and not self.inference:
            # print(f'\033[34mUsing Existed Label for scene:{scene_id}, l:{l_id}, r:{ann_id_r}, is_object:{is_object}\033[0m')
            labels = torch.tensor(np.load(label_path))
        elif not self.inference:
            # print(label_path)
            print(
                f'\033[31mWarning, No Existed Label for scene:{scene_id}, l:{l_id}, r:{ann_id_r}, is_object:{is_object}\033[0m')
            labels = self.match_points(
                ret_dict['l']['cloud'], ret_dict['r']['cloud'], l2r_pose, use_icp=use_icp,
                computing_device=self.matching_device)
            if not os.path.exists(self.label_dir):
                os.makedirs(self.label_dir)
            np.save(label_path, labels)
            labels = torch.from_numpy(labels)
        else:
            labels = None

        t3 = time.time()
        # print(f'loading time:{t2 -t1}, matching time:{t3 - t2}')

        ret_dict['labels'] = labels
        # print(labels.shape)
        if labels is not None:
            ret_dict['matches'] = get_matches(
                ret_dict, self.n_match, labels)  # shape (n_match, 4)
            if ret_dict['matches'] is None or not ret_dict['matches'].shape == (self.n_match, 2):
                f = open('err_log.txt', 'a')
                # print(labels)
                print(
                    f'\033[31mMatches is None for {scene_id} {l_id} {ann_id_r},{is_object}\033[0m')
                if ret_dict['matches'] is None:
                    f.write(
                        f'match_is_none,{self.camera},{scene_id},{l_id},{ann_id_r},{is_object}\n')
                else:
                    f.write(
                        f'match_shape_error,{self.camera},{scene_id},{l_id},{ann_id_r},{is_object}\n')
                f.close()
            # shape (n_non_match, 2) #TODO: n_non_match ?= n_match
            ret_dict['non_matches'] = get_non_matches(ret_dict, self.n_non_match)
            if not ret_dict['non_matches'].shape == (self.n_non_match, 2):
                f = open('err_log.txt', 'a')
                f.write(
                    f'non_match_shape_error,{self.camera},{scene_id},{l_id},{ann_id_r},{is_object},{is_object}\n')
                f.write(f'{scene_id} {l_id} {ann_id_r}\n')
                f.close()
        else:
            ret_dict['matches'] = torch.tensor([[0, 0]])
            ret_dict['non_matches'] = torch.tensor([[0, 1]])


        # Get the objectness samples

        # print(f'matches:{ret_dict["matches"].shape}, {ret_dict["matches"].dtype}')
        # print(f'non_matches:{ret_dict["non_matches"].shape}, {ret_dict["non_matches"].dtype}')

        idxs_aug_collection = {
            'l': None,
            'r': None
        }
        if self.use_augmentation:
            for lr in ['l', 'r']:
                # print("Before", lr)
                # print(ret_dict[lr]['coords'].shape, torch.unique(ret_dict[lr]['coords'], dim=0).shape)
                # print(64 * '-')

                cloud_aug, rotation_matrix_for_augmentation[lr] = self.augment_data(ret_dict[lr]['cloud'],
                                                                                    is_object=True or (is_object and (
                                                                                                lr == 'l')))
                if self.add_gaussian_noise_frequency >= 1 \
                        and np.random.uniform() < 1 / self.add_gaussian_noise_frequency:
                    d_ = np.random.uniform(0.5 * VOXEL_SIZE, 3.5 * VOXEL_SIZE)
                    cloud_aug = self.add_gaussian_noise(cloud_aug,
                                                        is_object=(is_object and (lr == 'l')),
                                                        density=d_)

                coords_aug = torch.tensor(np.ascontiguousarray(cloud_aug / self.voxel_size, dtype=np.int32))
                _, idxs_aug = ME.utils.sparse_quantize(coords_aug, return_index=True)

                coords_aug = coords_aug[idxs_aug]
                cloud_aug = cloud_aug[idxs_aug].float()
                objectness_label_aug = ret_dict[lr]['objectness_label'][idxs_aug]
                sem_seg_label_aug = ret_dict[lr]['sem_seg_label'][idxs_aug]

                idxs_aug_collection[lr] = idxs_aug
                ret_dict[lr]['cloud'] = cloud_aug
                ret_dict[lr]['idxs'] = ret_dict[lr]['idxs'][idxs_aug]
                ret_dict[lr]['coords'] = coords_aug
                ret_dict[lr]['rgb_selected'] = ret_dict[lr]['rgb_selected'][idxs_aug]
                ret_dict[lr]['objectness_label'] = objectness_label_aug
                ret_dict[lr]['sem_seg_label'] = sem_seg_label_aug

            matches_aug = []
            for i_m, match_pair in enumerate(ret_dict['matches']):
                v1 = (idxs_aug_collection['l'] == match_pair[0]).nonzero(as_tuple=True)[0]
                v2 = (idxs_aug_collection['r'] == match_pair[1]).nonzero(as_tuple=True)[0]
                if len(v1) > 0 and len(v2) > 0:
                    matches_aug.append(torch.tensor([v1[0], v2[0]]))
            if len(matches_aug) < 1:  # :TO DO FIND A BETTER WAY
                matches_aug.append(torch.tensor([0, 0]))
            if len(matches_aug) < self.n_match:
                matches_aug = matches_aug + (self.n_match - len(matches_aug)) * [torch.tensor([-1, -1])]
            matches_aug = torch.stack(matches_aug)

            non_matches_aug = []
            for i_m, non_match_pair in enumerate(ret_dict['non_matches']):
                v1 = (idxs_aug_collection['l'] == non_match_pair[0]).nonzero(as_tuple=True)[0]
                v2 = (idxs_aug_collection['r'] == non_match_pair[1]).nonzero(as_tuple=True)[0]
                if len(v1) > 0 and len(v2) > 0:
                    non_matches_aug.append(torch.tensor([v1[0], v2[0]]))
            if len(non_matches_aug) < 1:  # :TO DO FIND A BETTER WAY
                non_matches_aug.append(torch.tensor([0, 0]))
            if len(non_matches_aug) < self.n_non_match:  # if we have lost some points during the augmentation, fill it by -1
                non_matches_aug = non_matches_aug + (self.n_non_match - len(non_matches_aug)) * [torch.tensor([-1, -1])]
            non_matches_aug = torch.stack(non_matches_aug)

            ret_dict['matches'] = matches_aug
            ret_dict['non_matches'] = non_matches_aug

            rot_l = rotation_matrix_for_augmentation['l'].cpu().numpy()
            rot_r = rotation_matrix_for_augmentation['r'].cpu().numpy()
            T_l = np.identity(4)
            T_r = np.identity(4)
            T_l[:3, :3] = rot_l
            T_r[:3, :3] = rot_r
            l2r_pose = np.matmul(np.matmul(T_r.transpose(), l2r_pose), np.linalg.inv(T_l).transpose())

        if self.save_intermediate_product_frequency >= 1 and index % self.save_intermediate_product_frequency == 0:
            # DEBUG PURPOSE: save the middle product
            xyz_l, xyz_r = ret_dict['l']['cloud'], ret_dict['r']['cloud']
            rgb_l, rgb_r = ret_dict['l']['rgb_selected'], ret_dict['r']['rgb_selected']
            objectness_label = ret_dict['r']['objectness_label']
            sem_seg_label = ret_dict['r']['sem_seg_label']

            pcd_l = o3d.geometry.PointCloud()
            pcd_l.points = o3d.utility.Vector3dVector(xyz_l)
            pcd_l.colors = o3d.utility.Vector3dVector(rgb_l)

            pcd_r = o3d.geometry.PointCloud()
            pcd_r.points = o3d.utility.Vector3dVector(xyz_r)
            pcd_r.colors = o3d.utility.Vector3dVector(rgb_r)

            pcd_l.transform(l2r_pose)

            pcd_o = o3d.geometry.PointCloud()
            pcd_o.points = o3d.utility.Vector3dVector(xyz_r[objectness_label > 0.5])
            pcd_o.colors = o3d.utility.Vector3dVector(rgb_r[objectness_label > 0.5])

            pcd_s = o3d.geometry.PointCloud()
            pcd_s.points = o3d.utility.Vector3dVector(xyz_r[sem_seg_label > 0.5])
            pcd_s.colors = o3d.utility.Vector3dVector(rgb_r[sem_seg_label > 0.5])

            intermediate_product_path = \
                os.path.join("/DATA1/Benchmark/unseenpose/visualization", f'{self.camera}_{scene_id}_{l_id}_{ann_id_r}')

            if not os.path.exists(intermediate_product_path):
                os.makedirs(intermediate_product_path)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'left.ply'), pcd_l)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'right.ply'), pcd_r)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'objectness.ply'), pcd_o)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'semantic_segmentation.ply'), pcd_s)

        ret_dict['l2r_pose'] = torch.from_numpy(l2r_pose)
        ret_dict['repr'] = f'index: camera:{self.camera} scene_id:{scene_id}, l_id:{l_id}, ann_id_r:{ann_id_r}, is_object:{is_object}'
        # print("matches size in __getitem__ ", ret_dict['matches'])
        # print("non matches size in __getitem__ ", ret_dict['non_matches'])
        return ret_dict


def graspnet_collate_fn(batch):
    ret_dict = dict()
    for i, b in enumerate(batch):
        if b['matches'] == None:
            b['matches'] = torch.tensor([[]])
        if b['non_matches'] == None:
            b['non_matches'] = torch.tensor([[]])
    for lr in ['l', 'r']:
        # rgb = torch.stack([batch[i][lr]['rgb'] for i in range(len(batch))])
        num_points = torch.tensor([len(batch[i][lr]['idxs']) for i in range(len(batch))])
        idxs = torch.cat([batch[i][lr]['idxs'] for i in range(len(batch))])
        start_idx = torch.cumsum(num_points, dim=0) - num_points
        coords_list = [batch[i][lr]['coords'] for i in range(len(batch))]
        cloud_list = [torch.cat((batch[i][lr]['cloud'], batch[i][lr]['rgb_selected']), axis=1) for i in
                      range(len(batch))]
        objectness_list = [batch[i][lr]['objectness_label'].float() for i in range(len(batch))]
        sem_seg_list = [batch[i][lr]['sem_seg_label'].float() for i in range(len(batch))]

        labels_list = [torch.stack((objectness_list[i], sem_seg_list[i])).T for i in range(len(batch))]
        # print('obj ', objectness_list[0].sum())
        # print('sem seg ', sem_seg_list[0].sum())

        coords_batch, points_batch, labels_batch = ME.utils.sparse_collate(
            coords_list, cloud_list, labels_list)
        # print('=== DatapLoader Result ===')
        # print(f'num_points:{num_points.shape} {num_points.dtype}\nstart_idx:{start_idx.shape} {start_idx.dtype}')
        # print(f'coords_batch:{coords_batch.shape} {coords_batch.dtype}\npoints_batch:{points_batch.shape} {points_batch.dtype}\nidxs:{idxs.shape}{idxs.dtype}')

        ret_dict[lr] = {
            # 'rgb': rgb,
            'num_points': num_points,
            'start_idx': start_idx,
            'coords_batch': coords_batch.int(),  # ？
            'points_batch': points_batch.float(),
            'idxs': idxs,
            'objectness_label': labels_batch[:, 0],
            'sem_seg_label': labels_batch[:, 1]
        }

    ret_dict['repr'] = [batch[i]['repr'] for i in range(len(batch))]
    ret_dict['matches'] = torch.stack(
        [batch[i]['matches'] for i in range(len(batch))])
    ret_dict['non_matches'] = torch.stack(
        [batch[i]['non_matches'] for i in range(len(batch))])
    ret_dict['l2r_pose'] = torch.stack(
        [batch[i]['l2r_pose'] for i in range(len(batch))])

    # print(f'matches:{ret_dict["matches"].shape}, {ret_dict["matches"].dtype}')
    # print(f'non_matches:{ret_dict["non_matches"].shape}, {ret_dict["non_matches"].dtype}')

    # ret_dict['matches'] = torch.stack([batch[i]['matches'] for i in range(len(batch))])
    # ret_dict['non_matches'] = torch.stack([batch[i]['non_matches'] for i in range(len(batch))])
    # print("matches size in __collatefn__ ", ret_dict['matches'].shape)
    # print("non matches size in __collatefn__ ", ret_dict['non_matches'].shape)
    return ret_dict


def get_graspnet_dataloader(root, camera='realsense', split='all', voxel_size=VOXEL_SIZE):
    train_dataset = GraspNetDataset(
        root=GRASPNET_ROOT,
        camera=camera,
        split=split,
        list_file_dir='graspnet_offline_list',
        label_dir=LABEL_DIR,
        voxel_size=voxel_size
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=graspnet_collate_fn
    )
    return dataloader


if __name__ == "__main__":
    dataloader = GraspNetDataset('/disk1/graspnet')
    # rgb_l, rgb_r, cloud_l, cloud_r = dataloader.__getitem__(0,0,1)

    # rgb_l, cloud_l = dataloader.load_rgb_with_pointcloud(scene_id,ann_id_l)
    # rgb_r, cloud_r = dataloader.load_rgb_with_pointcloud(scene_id,ann_id_r)

    # poses = np.load(dataloader.ann_posespath[scene_id])
    # pose_l = poses[ann_id_l]
    # pose_r = poses[ann_id_r]
#     label = self.match_points(cloud_l, cloud_r, pose_l, pose_r)

#         # cloud_l = torch.tensor(np.transpose(cloud_l, (2, 0, 1)))
#         # cloud_r = torch.tensor(np.transpose(cloud_r, (2, 0, 1)))
#         cloud_l = torch.tensor(cloud_l.T)
#         cloud_r = torch.tensor(cloud_r.T)

#         return rgb_l,rgb_r, cloud_l, cloud_r, label


# if __name__ == "__main__":
#     dataloader = GraspNetDataset(GRASPNET_ROOT)
#     begin = time.clock()
#     rgb_l,rgb_r, cloud_l, cloud_r, label = dataloader.__getitem__(0,0,1)
#     end = time.clock()
#     print()
