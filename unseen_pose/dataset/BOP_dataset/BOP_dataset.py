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
import imageio

from transforms3d.axangles import axangle2mat
from graspnetAPI.utils.utils import create_point_cloud_from_depth_image, CameraInfo, parse_posevector
from graspnetAPI.utils.xmlhandler import xmlReader
from unseen_pose.utils import get_posed_points, torch_get_closest_point, convert_data_to_device
from unseen_pose.constant import GRASPNET_ROOT, GRASPNET_ROOT_REAL, RANDOM_SEED, MODEL_DOWNSAMPLED_DIR, LABEL_DIR, \
    VOXEL_SIZE, MODEL_DOWNSAMPLED_DIR_REAL
from unseen_pose.dataset.BOP_dataset.BOP_constant import PAIR_NUM, SCENE_PAIR_NUM, ANN_NUM, RESIZE_WIDTH, \
    RESIZE_HEIGHT, ORIGIN_HEIGHT, ORIGIN_WIDTH, CLOSE_DIST_THRES, REMOTE_DIST_THRES
from unseen_pose.dataset.graspnet.graspnet_dataset import get_matches, get_non_matches


logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class BOPDataset(Dataset):

    def __init__(self, root, dataset_name='ycbv', split='all', voxel_size=VOXEL_SIZE, n_match=10, n_non_match=10,
                 model_pair_ratio=1.0, scene_pair_ratio=1.0, list_file_dir='bop_offline_list', label_dir=None,
                 inference=False, matching_devices='cpu', generate_label=False, use_remove_mask=True, use_augmentation=True,
                 mask_switch_frequency=3, add_gaussian_noise_frequency=3, save_intermediate_product_frequency=1000):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.voxel_size = voxel_size
        self.n_match = n_match
        self.n_non_match = n_non_match
        self.label_dir = label_dir
        self.inference = inference
        self.matching_device = matching_devices
        self.generate_label = generate_label
        self.model_pair_ratio = model_pair_ratio
        self.scene_pair_ratio = 0.0
        self.use_remove_mask = use_remove_mask
        self.use_augmentation = use_augmentation
        self.mask_switch_frequency = mask_switch_frequency
        self.add_gaussian_noise_frequency = add_gaussian_noise_frequency
        self.save_intermediate_product_frequency = save_intermediate_product_frequency

        split_name_dict = {
            'ycbv': 'train_real',
            'tless': 'test_primesense'
        }

        self.split_name = split_name_dict[self.dataset_name]

        print(self.matching_device, torch.cuda.device_count())

        assert dataset_name in ['ycbv', 'tless'], 'The dataset required is not supported'
        if dataset_name == 'ycbv':
            if split == 'all':
                self.sceneIds = [65, 69, 72] + \
                                [4, 5, 9, 10, 18, 20, 21, 25, 29, 32, 36, 42, 43, 47, 53, 54, 57, 60, 62, 64, 66, 73,
                                 77, 81, 82, 88, 89]
            elif split == 'train':
                self.sceneIds = [65, 69, 72]
            elif split == 'test':
                self.sceneIds = [4, 5, 9, 10, 18, 20, 21, 25, 29, 32, 36, 42, 43, 47, 53, 54, 57, 60, 62, 64, 66, 73,
                                 77, 81, 82, 88, 89]
        elif dataset_name == 'tless':
            if split == 'all':
                self.sceneIds = list(range(1, 21))
            elif split == 'train':
                self.sceneIds = list(range(1, 2))
            elif split == 'test':
                self.sceneIds = list(range(1, 21))
        else:
            raise Exception('The dataset required is not supported')

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
            self.scene_pair_list = self.scene_pair_list[:selected_total_scene_pair_num, :3]
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
            self.model_pair_list = self.model_pair_list[:selected_total_model_pair_num]
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

        self.sceneIds = ['{}'.format(
            str(x).zfill(6)) for x in self.sceneIds]
        # print(self.sceneIds)
        self.colorpath = []
        self.depthpath = []
        self.metapath = []
        self.maskpath = []
        self.scenename = []
        self.frameid = []
        self.ann_posespath = []

        resize = transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH))
        totensor = transforms.ToTensor()
        trans_list = [resize, totensor]
        self.rgb_transform = transforms.Compose(trans_list)
        with open(os.path.join(root, dataset_name, self.split_name, self.sceneIds[0], 'scene_camera.json')) as f:
            self.camera_poses = json.load(f)
        self.intrinsic = np.array(list(self.camera_poses.values())[0]['cam_K']).reshape(3, 3)
        self.factor_depth = list(self.camera_poses.values())[0]["depth_scale"]

        # print(self.intrinsic)
        # print(self.factor_depth)
        self.ann_num_per_scene = {}

        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            ann_num = len([f for f in os.listdir(os.path.join(root, dataset_name, self.split_name, x, 'rgb'))])
            self.ann_num_per_scene[x] = ann_num
            for img_num in range(ann_num):
                self.colorpath.append(os.path.join(
                    root, dataset_name, self.split_name, x, 'rgb', str(img_num).zfill(6) + '.png'))
                self.depthpath.append(os.path.join(
                    root, dataset_name, self.split_name, x, 'depth', str(img_num).zfill(6) + '.png'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

    def __len__(self):
        if self.from_list:
            return len(self.all_pair_list)
        else:
            return sum(list(self.ann_num_per_scene.values()))

    def load_rgb_with_pointcloud(self, scene_id, ann_or_obj_id, is_object):
        '''
        return:
        rgb: torch.tensor [3, RESIZE_HEIGHT, RESIZE_WIDTH] for scene,
            (n, 3) for object.
        cloud: torch.tensor [3, RESIZE_HEIGHT, RESIZE_WIDTH] for scene,
            (n, 3) for object.
        '''
        if is_object:
            pcd = o3d.io.read_point_cloud(
                os.path.join(self.root, self.dataset_name, 'models_down', '%03d.ply' % ann_or_obj_id))
            xyz = np.asarray(pcd.points)
            if self.dataset_name in ['ycbv', 'tless']:
                xyz = 0.001 * xyz
            pcd.points = o3d.utility.Vector3dVector(xyz)
            # print(os.path.join(MODEL_DOWNSAMPLED_DIR, '%03d.ply' % ann_or_obj_id))
            rgb = torch.from_numpy(np.asarray(pcd.colors))  # (n, 3)
            cloud = torch.from_numpy(np.asarray(pcd.points))  # (n, 3)
        else:
            # /DATA3/Benchmark/BOP_datasets/ycbv/train_real/000054/rgb
            color_path = os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                      scene_id, 'rgb', '%06d.png' % ann_or_obj_id)
            depth_path = os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                      scene_id, 'depth', '%06d.png' % ann_or_obj_id)
            # index = scene_id * ANN_NUM + ann_id
            rgb = Image.open(color_path)
            rgb = self.rgb_transform(rgb)
            depth = Image.open(depth_path)
            depth = np.array(depth.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.NEAREST))

            if self.dataset_name in ['tless']:
                with open(os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                                                            scene_id, 'scene_camera.json')) as f:
                    camera_poses = json.load(f)
                intrinsic = np.array(camera_poses[str(ann_or_obj_id)]['cam_K']).reshape(3, 3)
                factor_depth = 1000 / camera_poses[str(ann_or_obj_id)]['depth_scale']
            else:
                intrinsic = self.intrinsic
                factor_depth = 1000 / self.factor_depth

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

    def get_obj_6dpose(self, scene_id, ann_id, obj_id, dataset_name):
        with open(os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                                                           scene_id, 'scene_gt.json')) as fp:
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

        mask_path = os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                 scene_id, 'label', '%06d.png' % ann_id)
        if not os.path.exists(mask_path):
            mask_visib_dir = os.path.join(self.root, self.dataset_name, self.split_name, '%06d' % scene_id, 'mask_visib')
            with open(os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                                                               scene_id, 'scene_gt.json')) as fp:
                scene_gt_dict = json.load(fp)
            ann = scene_gt_dict[str(ann_id)]

            mask = np.zeros((ORIGIN_HEIGHT, ORIGIN_WIDTH))

            for i_obj, obj_notation in enumerate(ann):
                mask_visib_path = os.path.join(mask_visib_dir, '%06d_%06d.png'%(ann_id, i_obj))
                if os.path.exists(mask_visib_path):
                    obj_id = obj_notation['obj_id']
                    mask_visib_obj = cv2.imread(mask_visib_path, cv2.IMREAD_UNCHANGED)
                    mask[mask_visib_obj > 0] = obj_id + 1
            if not os.path.exists(os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                                                                       scene_id, 'label')):
                os.makedirs(os.path.join(self.root, self.dataset_name, self.split_name, '%06d' %
                                         scene_id, 'label'))
            imageio.imwrite(mask_path, mask.astype(np.uint16))

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask[:, :], (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)
        return torch.from_numpy(mask)

    def __getitem__(self, *args):
        '''
        :param args: support two types get item, index or
         scene_num, ann_id_l, ann_id_r, is_object
        :return:
        a dict of info
        '''
        import time
        t1 = time.time()
        assert len(args) == 1, 'only support index attaching'

        index = args[0]
        if self.from_list:
            scene_id, l_id, ann_id_r = self.all_pair_list[index]
            is_object = self.model_pair_mask[index]
        else:
            scene_id, l_id, ann_id_r = self.get_ids(index)
            raise NotImplementedError('not from list not implemented')

        # print(
        #     f'calling index: dataset name:{self.dataset_name}' +
        #     f' scene_id:{scene_id},l_id:{l_id},ann_id_r:{ann_id_r},is_object:{is_object}')
        assert is_object, 'Only assert object-scene registration'

        label_path = os.path.join(self.label_dir, '%s-%03d-%03d-%04d-object.npy' %
                                  (self.dataset_name, scene_id, l_id, ann_id_r))

        if self.generate_label:
            if os.path.exists(label_path):
                print(f'Using existed Label {label_path}')
                return 0

        ret_dict = {}

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
            # print(rgb.shape, cloud.shape)

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

            if sem_seg_label is not None and sem_seg_label.sum() <= 0:
                f = open(f'err_log.txt', 'a')
                print(
                    f'\033[31mMatches is None for {scene_id} {l_id} {ann_id_r},{is_object}\033[0m')
                f.write(
                    f'match_is_none,{self.dataset_name},{scene_id},{l_id},{ann_id_r},{is_object}\n')
                f.close()

            if is_object and (lr == 'l'):
                rgb_selected = rgb[idxs]  # (n, 3)
            else:
                rgb_selected = rgb.permute(1, 2, 0).reshape((-1, 3))[idxs]

            # mask_ = sem_seg_mask if (index % self.mask_switch_frequency) != 0 else objectness_mask
            mask_ = objectness_mask
            ret_dict[lr] = {'rgb': rgb, 'idxs': idxs, 'coords': coords, 'rgb_selected': rgb_selected,
                            'cloud': cloud, 'origin_cloud': origin_cloud, 'mask': mask_,
                            'objectness_label': objectness_label, 'sem_seg_label': sem_seg_label}

        # poses = self.camera_poses
        # pose_r = poses[ann_id_r]

        obj_pose = self.get_obj_6dpose(scene_id, ann_id_r, l_id, self.dataset_name)
        l2r_pose = obj_pose
        use_icp = False

        if os.path.exists(label_path) and not self.inference:
            # print(f'\033[34mUsing Existed Label for scene:{scene_id}, l:{l_id}, r:{ann_id_r}, is_object:{is_object}\033[0m')
            labels = torch.tensor(np.load(label_path))
        elif not self.inference:
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

        if labels is not None:
            ret_dict['labels'] = labels
            # print(labels.shape)
            ret_dict['matches'] = get_matches(
                ret_dict, self.n_match, labels)  # shape (n_match, 4)
            if ret_dict['matches'] is None or not ret_dict['matches'].shape == (self.n_match, 2):
                f = open('err_log.txt', 'a')
                # print(labels)
                print(
                    f'\033[31mMatches is None for {scene_id} {l_id} {ann_id_r},{is_object}\033[0m')
                if ret_dict['matches'] is None:
                    f.write(
                        f'match_is_none,{self.dataset_name},{scene_id},{l_id},{ann_id_r},{is_object}\n')
                else:
                    f.write(
                        f'match_shape_error,{self.dataset_name},{scene_id},{l_id},{ann_id_r},{is_object}\n')
                f.close()
            ret_dict['non_matches'] = get_non_matches(ret_dict, self.n_non_match)
            if not ret_dict['non_matches'].shape == (self.n_non_match, 2):
                f = open('err_log.txt', 'a')
                f.write(
                    f'non_match_shape_error,{self.dataset_name},{scene_id},{l_id},{ann_id_r},{is_object},{is_object}\n')
                f.write(f'{scene_id} {l_id} {ann_id_r}\n')
                f.close()
        else:
            ret_dict['matches'] = torch.tensor([[0, 0]])
            ret_dict['non_matches'] = torch.tensor([[0, 1]])

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

                # print("After", lr)
                # print(ret_dict[lr]['coords'].shape, torch.unique(ret_dict[lr]['coords'], dim=0).shape)
                # print(64 * '-')
                # print(64 * "=")
            matches_aug = []
            for i_m, match_pair in enumerate(ret_dict['matches']):
                v1 = (idxs_aug_collection['l'] == match_pair[0]).nonzero(as_tuple=True)[0]
                v2 = (idxs_aug_collection['r'] == match_pair[1]).nonzero(as_tuple=True)[0]
                if len(v1) > 0 and len(v2) > 0:
                    matches_aug.append(torch.tensor([v1[0], v2[0]]))
            if len(matches_aug) < 1:  # :TO DO FIND A BETTER WAY
                matches_aug.append(torch.tensor([0, 0]))
            matches_aug = torch.stack(matches_aug)

            non_matches_aug = []
            for i_m, non_match_pair in enumerate(ret_dict['non_matches']):
                v1 = (idxs_aug_collection['l'] == non_match_pair[0]).nonzero(as_tuple=True)[0]
                v2 = (idxs_aug_collection['r'] == non_match_pair[1]).nonzero(as_tuple=True)[0]
                if len(v1) > 0 and len(v2) > 0:
                    non_matches_aug.append(torch.tensor([v1[0], v2[0]]))
            if len(non_matches_aug) < 1:  # :TO DO FIND A BETTER WAY
                non_matches_aug.append(torch.tensor([0, 0]))
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
        ret_dict['l2r_pose'] = torch.from_numpy(l2r_pose)

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
                os.path.join(f"/DATA3/Benchmark/BlenderProcGoogle1000/{self.dataset_name}/visualization", f'{scene_id}_{l_id}_{ann_id_r}')

            if not os.path.exists(intermediate_product_path):
                os.makedirs(intermediate_product_path)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'left.ply'), pcd_l)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'right.ply'), pcd_r)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'objectness.ply'), pcd_o)
            o3d.io.write_point_cloud(os.path.join(intermediate_product_path, 'semantic_segmentation.ply'), pcd_s)

        ret_dict['repr'] = f'index: dataset_name:{self.dataset_name} scene_id:{scene_id}, l_id:{l_id}, ann_id_r:{ann_id_r}, is_object:{is_object}'
        return ret_dict


if __name__ == "__main__":
    dataset = BOPDataset('/DATA3/Benchmark/BOP_datasets')
    print(dataset)
