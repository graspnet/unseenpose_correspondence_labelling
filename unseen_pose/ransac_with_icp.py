import numpy as np
import open3d as o3d
import random
import torch
import time
import open3d as o3d
from tqdm import tqdm
import os
import torch.nn as nn

from graspnetAPI import GraspNetEval
from graspnetAPI.utils.utils import create_point_cloud_from_depth_image,  CameraInfo

from .dataset.graspnet.graspnet_constant import RESIZE_HEIGHT, RESIZE_WIDTH, ORIGIN_HEIGHT, ORIGIN_WIDTH
from .utils import torch_get_closest_point
from .vis import vis_match_pairs, draw_line
from .constant import GRASPNET_ROOT
from .dataset.graspnet.graspnet_dataset import get_graspnet_dataloader, GraspNetDataset
from .dataset.graspnet.graspnet_dataset import convert_data_to_device
from .dataset.graspnet.graspnet_dataset import graspnet_collate_fn


def solve_transformation_matrix(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t
    T[-1, -1] = 1.0
    return T


def ransac(object_cloud, target_cloud, object_points, target_points, thresh=0.02, maxIteration=300, vis=False, inlier_method='match'):
    '''
    object/target cloud: numpy arrary of (HEIGHT * WIDTH,3)
    object/target points: numpy array of (n,3)
    thresh: Threshold distance from the plane which is considered inlier.
    minPoints: iter begin point number
    maxIteration: max number of iterations
    inlier_method: use matches or cloud to calculate inlier

    :return: 6d pose of the object
    '''
    print(f'calling with')
    print(f'object_cloud.shape:{object_cloud.shape}')
    print(f'target_cloud.shape:{target_cloud.shape}')
    print(f'object_points.shape:{object_points.shape}')
    print(f'target_points.shape:{target_points.shape}')
    device = 'cuda:0'
    N = object_points.shape[0]
    best_pose = np.eye(4, dtype=float)
    best_inliers = []
    best_indices = []
    object_points = object_points.numpy()
    target_points = target_points.numpy()
    start_time = time.time()
    for _ in tqdm(range(maxIteration), 'RANSAC iterations'):
        id_samples = random.sample(range(N), min(20, N))
        obj_samples = object_points[id_samples]
        target_samples = target_points[id_samples]

        pose = solve_transformation_matrix(obj_samples, target_samples)

        inliers = []
        if inlier_method == 'match':
            obj_to_tar = np.matmul(pose, np.column_stack(
                (object_points, np.ones((object_points.shape[0], 1)))).T).T
            obj_to_tar = np.delete(obj_to_tar, 3, axis=1)
            assert obj_to_tar.shape == target_points.shape
            min_dist = np.sqrt(
                np.sum(np.square(obj_to_tar - target_points), axis=1))
            inliers = np.where(min_dist <= thresh)[0]

        elif inlier_method == 'cloud':
            obj_to_tar = np.matmul(pose, np.column_stack((object_cloud,np.ones((object_cloud.shape[0],1)))).T).T
            obj_to_tar = np.delete(obj_to_tar, 3, axis = 1)
            obj_to_tar_tensor = torch.tensor(obj_to_tar).to(device)
            target_cloud_tensor = torch.tensor(target_cloud).to(device)
            min_dist, indices = torch_get_closest_point(obj_to_tar_tensor, target_cloud_tensor)
            inliers = torch.where(min_dist<=thresh)[0]
            if len(inliers) > len(best_inliers):
                best_indices = indices[inliers]
        else:
            raise ValueError('Unknown inlier method')

        if len(inliers) > len(best_inliers):
            best_pose = pose
            best_inliers = inliers
        # if len(best_inliers) > object_cloud.shape[0] * 0.5:
        #     break
    end_time = time.time()
    if inlier_method == 'match':
        inlier_ratio = len(best_inliers) / object_points.shape[0]
    elif inlier_method == 'cloud':
        inlier_ratio = len(best_inliers) / object_cloud.shape[0]
    else:
        raise ValueError(f'Unknown inlier_method:{inlier_method}')

    if inlier_method == 'match':
         best_inlier_l = object_points[best_inliers]
         best_inlier_r = target_points[best_inliers]
    elif inlier_method == 'cloud':
         best_inlier_l = object_cloud[best_inliers]
         best_inlier_r =target_cloud[best_indices]
    else:
        raise ValueError(f'Unknown inlier_method:{inlier_method}')


    if(len(best_inliers) > 0):
        best_pose = solve_transformation_matrix(best_inlier_l, best_inlier_r)

    print('best rate of inliner is %f' % (inlier_ratio, ))
    print('the time of this function is %f' % (end_time - start_time))
    # return torch.tensor(np.linalg.inv(best_pose))
    return torch.tensor(best_pose)


def down_sampling(cloud, voxel_size = 0.02):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down_pcd.points)


def ransac_with_dataloader(dict, mapping, thresh=0.005, maxIteration=1000):
    '''
    depth:(RESIZE_HEIGHT, RESIZE_WIDTH)
    map: (n,2) tensor
    '''
    # sx = RESIZE_WIDTH / ORIGIN_WIDTH
    # sy = RESIZE_HEIGHT / ORIGIN_HEIGHT
    # camera = CameraInfo(RESIZE_WIDTH, RESIZE_HEIGHT, sx * intrinsic[0][0], sy * intrinsic[1][1], sx * intrinsic[0][2] + 0.5 * sx - 0.5, sy * intrinsic[1][2] + 0.5 * sy - 0.5, factor_depth)

    # cloud_l = create_point_cloud_from_depth_image(depth_l, camera, organized=True) #(w*h,3)
    # cloud_r = create_point_cloud_from_depth_image(depth_r, camera, organized=True) #(w*h,3)

    cloud_l = dict['l']['cloud']
    cloud_r = dict['r']['cloud']

    index_l = mapping[:, 0]
    index_r = mapping[:, 1]
    obj_points = cloud_l[index_l]
    target_points = cloud_r[index_r]
    # print(obj_points.shape)
    # print(target_points.shape)

    best_pose = ransac(obj_points, target_points, thresh, maxIteration)
    return best_pose


def vis_ransac(path):
    data = np.load(path)
    cloud_l = data['cloud_l']
    cloud_r = data['cloud_r']
    match = data['matches']
    pose = data['pose']
    # pose = np.array([[-0.75100243, -0.53396846, -0.38842377, 0.07752744], [0.46230121, -0.00518938, -0.88670776, 0.4461221],[0.4714583, -0.84548847, 0.25075151, 0.35481029], [0., 0., 0., 1.]])
    rgb_l = data['color_l']
    rgb_r = data['color_r']

    expand_l = np.column_stack((cloud_l, np.ones((cloud_l.shape[0], 1))))
    cloud_l = np.delete(np.matmul(pose, expand_l.T).T, 3, axis=1)

    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(cloud_l)
    pcd_l.colors = o3d.utility.Vector3dVector(rgb_l.reshape((3, -1)).T)
    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(cloud_r)
    pcd_r.colors = o3d.utility.Vector3dVector(rgb_r.reshape((3, -1)).T)

    num_pair = min(10, match.shape[0])
    lines = []
    indices = random.sample(range(match.shape[0]), num_pair)
    for i in range(num_pair):
        l_id = match[indices[i]][0]
        r_id = match[indices[i]][1]

        l, v1, v2 = draw_line(cloud_l[l_id], cloud_r[r_id])
        lines += [l, v1, v2]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines, frame])


def preprocess_batch_data(batch_data, matches):
    '''
    **Input:**

    - batch_data: dict returned by data loader with batch_size == 1

    - matches: torch.tensor of shape (n, 2)

    **Output:**

    - data needed by ransac function.
    '''
    assert batch_data['l']['num_points'].shape[0] == 1, 'batch_size must be 1 for inference'
    data = dict()
    for lr_id, lr in enumerate(['l', 'r']):
        data[lr] = dict()
        data[lr]['match_index'] = matches[:, lr_id]
        data[lr]['cloud'] = batch_data[lr]['points_batch'][:, :3]
        # data[lr]['points_idxs'] = batch_data[lr]['idxs']
        # data[lr]['match_in_points_idx'] = torch.searchsorted(data[lr]['points_idxs'], data[lr]['match_index'])
        data[lr]['match_points'] = data[lr]['cloud'][data[lr]['match_index']]
    return down_sampling(data['l']['cloud']), down_sampling(data['r']['cloud']), data['l']['match_points'], data['r']['match_points']


def vis_matches(batch_data, matches, vis=True):
    '''
    **Input:**

    - batch_data: dict returned by data loader with batch_size == 1

    - matches: torch.tensor of shape (n, 2)

    **Output:**

    - Visualize result.
    '''
    rgbxyz_l = batch_data['l']['points_batch'].detach().numpy().copy()
    rgbxyz_r = batch_data['r']['points_batch'].detach().numpy().copy()
    rgb_l = rgbxyz_l[:,3:]
    points_l = rgbxyz_l[:,:3]
    rgb_r = rgbxyz_r[:,3:]
    points_r = rgbxyz_r[:,:3]
    # rgb_l = batch_data['l']['rgb'][0].detach().numpy().copy().transpose(
    #     (1, 2, 0)).reshape((-1, 3))  # (288 * 384, 3)
    # rgb_r = batch_data['r']['rgb'][0].detach().numpy().copy().transpose(
    #     (1, 2, 0)).reshape((-1, 3))  # (288 * 384, 3)
    # points_l = batch_data['l']['points_batch'].detach(
    # ).numpy().copy()  # (n, 3)
    # points_r = batch_data['r']['points_batch'].detach(
    # ).numpy().copy()  # (n, 3)
    idx_l = matches[:, 0].detach().numpy().copy()
    idx_r = matches[:, 1].detach().numpy().copy()
    # match_idx_l = torch.searchsorted(
    #     idx_l, matches[:, 0]).detach().numpy().copy()
    # match_idx_r = torch.searchsorted(
    #     idx_r, matches[:, 1]).detach().numpy().copy()

    points_r[:, 1] = points_r[:, 1] + 0.6
    points_r[:, 2] = points_r[:, 2] + 0.6

    pcd_l = o3d.geometry.PointCloud()
    # print(f'points_l.shape:{points_l}, rgb.shape:{rgb_l[idx_l].shape}')
    pcd_l.points = o3d.utility.Vector3dVector(points_l)
    pcd_l.colors = o3d.utility.Vector3dVector(rgb_l)

    pcd_r = o3d.geometry.PointCloud()
    # print(f'points_r.shape:{points_r}, rgb.shape:{rgb_r[idx_r].shape}')
    pcd_r.points = o3d.utility.Vector3dVector(points_r)
    pcd_r.colors = o3d.utility.Vector3dVector(rgb_r)
    lines = []
    for i in range(len(idx_l)):
        l_id = idx_l[i]
        r_id = idx_r[i]
        l, v1, v2 = draw_line(points_l[l_id], points_r[r_id])
        lines += [l, v1, v2]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    if vis:
        o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines, frame])
    return pcd_l, pcd_r.translate(np.array([0, -0.6, -0.6]))

def get_tsne_pcd(feat, batch_data, using_cuda = True):
    # if using_cuda:
    #     from tsnecuda import TSNE
    # else:
    from sklearn.manifold import TSNE
    feature_l = nn.functional.normalize(feat['l']['feature_tsne'], dim = 1)
    feature_r = nn.functional.normalize(feat['r']['feature_tsne'], dim = 1)
    num_l = len(feature_l)
    num_r = len(feature_r)
    feature_all = torch.cat((feature_l, feature_r)).cpu().detach().numpy().copy()
    tsne_feature = np.empty((num_l + num_r, 3), dtype=float)
    tsne_feature[:, :3] = TSNE(n_components=3).fit_transform(feature_all)
    # tsne_feature[:, 2] = tsne_feature[:, 1]
    tsne_feature = (tsne_feature - tsne_feature.min()) / (tsne_feature.max() - tsne_feature.min())
    tsne_color_l = tsne_feature[:num_l]

    tsne_color_r = tsne_feature[num_l:]
    # points_l = batch_data['l']['points_batch'][:, :3].cpu().detach().numpy().copy()
    points_l = feat['l']['rgbxyz_tsne'][:, :3].cpu().detach().numpy().copy()
    # idxs_l = feat['l']['idx'].cpu().detach().numpy().copy()

    # idxs_r = feat['r']['idx'].cpu().detach().numpy().copy()
    points_r = feat['r']['rgbxyz_tsne'][:, :3].cpu().detach().numpy().copy()

    print(f'points_l.shape:{points_l.shape}')
    print(f'colors_l.shape:{tsne_color_l.shape}')

    print(f'points_r.shape:{points_r.shape}')
    print(f'colors_r.shape:{tsne_color_r.shape}')

    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(points_l)
    pcd_l.colors = o3d.utility.Vector3dVector(tsne_color_l)
    # o3d.visualization.draw_geometries([pcd_l])
    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(points_r)
    pcd_r.colors = o3d.utility.Vector3dVector(tsne_color_r)
    # o3d.visualization.draw_geometries([pcd_r])
    return pcd_l, pcd_r


def vis_inference(batch_data, matches, feat, using_tsne = True, vis_match=True, vis_ransac=True):
    '''
    **Input:**

    - batch_data: dict returned by data loader with batch_size == 1

    - matches: torch.tensor of shape (n, 2)

    **Output:**

    - Visualize result.
    '''
    
    assert batch_data['l']['num_points'].shape[0] == 1, 'batch_size must be 1 for inference'
    rgb_pcd_l, rgb_pcd_r = vis_matches(batch_data, matches, vis=vis_match)
    tsne_pcd_l, tsne_pcd_r = get_tsne_pcd(feat, batch_data)
    if using_tsne:
        pcd_l = tsne_pcd_l
        pcd_r = tsne_pcd_r
    else:
        pcd_l = rgb_pcd_l
        pcd_r = rgb_pcd_r
    if vis_ransac:
        cloud_l, cloud_r, match_points_l, match_points_r = preprocess_batch_data(
            batch_data, matches)
        pose = ransac(cloud_l, cloud_r, match_points_l, match_points_r)
        pcd_l.transform(pose)
        # idx_l = batch_data['l']['idxs']
        # idx_r = batch_data['r']['idxs']
        match_idx_l = matches[:, 0]
        match_idx_r = matches[:, 1]
        # match_idx_l = torch.searchsorted(
        #     idx_l, matches[:, 0]).detach().numpy().copy()
        # match_idx_r = torch.searchsorted(
        #     idx_r, matches[:, 1]).detach().numpy().copy()

        lines = []
        for i in range(len(match_idx_l)):
            l_id = match_idx_l[i]
            r_id = match_idx_r[i]
            l, v1, v2 = draw_line(np.asarray(pcd_l.points)[l_id], np.asarray(pcd_r.points)[r_id])
            lines += [l, v1, v2]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines, frame])


def test_ransac_with_dataloader(scene_id, ann_id_l, ann_id_r, n_match, n_non_match, dump=True, vis=True):
    if n_non_match == 0:
        load_n_non_match = 10
    else:
        load_n_non_match = n_non_match
    dataset = GraspNetDataset(GRASPNET_ROOT, n_match=n_match,
                              n_non_match=load_n_non_match, label_dir='graspnet_labels')
    item_dict = dataset.__getitem__(scene_id, ann_id_l, ann_id_r)
    batch_data = graspnet_collate_fn([item_dict])
    matches = item_dict['matches']
    if n_non_match > 0:
        matches = torch.cat((matches, item_dict['non_matches']), 0)
    pose = ransac(*(preprocess_batch_data(batch_data, matches)))
    print(f'pose:{pose}')
    if dump:
        if not os.path.exists('vis'):
            os.makedirs('vis')
        cloud_l = item_dict['l']['origin_cloud']
        cloud_r = item_dict['r']['origin_cloud']
        np.savez('vis/ransac.npz', cloud_l=cloud_l, cloud_r=cloud_r,
                 color_l=item_dict['l']['rgb'], color_r=item_dict['r']['rgb'], matches=matches, pose=pose)
    if vis:
        vis_ransac('vis/ransac.npz')


def test_ransac_with_label(scene_id, ann_id_l, ann_id_r, n_match, n_non_match, dump=True, vis=True):
    if n_non_match == 0:
        load_n_non_match = 10
    else:
        load_n_non_match = n_non_match
    dataset = GraspNetDataset(GRASPNET_ROOT, n_match=n_match,
                              n_non_match=load_n_non_match, label_dir='graspnet_labels')
    item_dict = dataset.__getitem__(scene_id, ann_id_l, ann_id_r)
    # vis_match_pairs(item_dict)

    cloud_l = item_dict['l']['origin_cloud']
    cloud_r = item_dict['r']['origin_cloud']
    matches = item_dict['matches']
    if n_non_match > 0:
        matches = torch.cat((matches, item_dict['non_matches']), 0)

    points_l = cloud_l[matches[:, 0]]
    points_r = cloud_r[matches[:, 1]]

    cloud_l_down_sampled = down_sampling(cloud_l, 0.001)
    cloud_r_down_sampled = down_sampling(cloud_r, 0.001)
    print('cloud_down_sampled shape:', cloud_l_down_sampled.shape, cloud_r_down_sampled.shape )
    fake_data = dict()
    fake_data['matches'] = matches
    fake_data['l'] = dict()
    fake_data['r'] = dict()
    fake_data['l']['origin_cloud'] = cloud_l
    fake_data['r']['origin_cloud'] = cloud_r
    fake_data['l']['rgb'] = item_dict['l']['rgb']
    fake_data['r']['rgb'] = item_dict['r']['rgb']
    vis_match_pairs(fake_data)

    pose = ransac(cloud_l_down_sampled, cloud_r_down_sampled, points_l,points_r, inlier_method='cloud')
    # icp(cloud_l, cloud_r, pose)
    print(f'pose:{pose}')
    if dump:
        if not os.path.exists('vis'):
            os.makedirs('vis')
        np.savez('vis/ransac.npz', cloud_l=cloud_l, cloud_r=cloud_r,
                 color_l=item_dict['l']['rgb'], color_r=item_dict['r']['rgb'], matches=matches, pose=pose)
    if vis:
        vis_ransac('vis/ransac.npz')
    

def icp(obj_cloud, tar_cloud, threshold = 0.01):
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(obj_cloud)
    
    pcd_tar = o3d.geometry.PointCloud()
    pcd_tar.points = o3d.utility.Vector3dVector(tar_cloud)

    pcd_obj.remove_radius_outlier(nb_points=16, radius=0.05)
    pcd_tar.remove_radius_outlier(nb_points=16, radius=0.05)

    center_obj = pcd_obj.get_center()
    center_tar = pcd_obj.get_center()

    pose = np.eye(4)
    pose[:3, :] = (center_tar - center_obj).reshape((3, 1))
    pcd_obj.transform(pose)
    evaluation = o3d.pipelines.registration.registration_icp(pcd_obj, pcd_tar, threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(evaluation.transformation)

    return pcd_obj, pcd_tar
