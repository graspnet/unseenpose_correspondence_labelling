import open3d as o3d
import numpy as np
from graspnetAPI import GraspNet, GraspNetEval
import os
import torch
import tqdm
import MinkowskiEngine as ME
from .constant import KEY_POINT_NUM, MODEL_KEY_POINT_DIR, GRASPNET_ROOT, LABEL_DIR
# from .dataset.graspnet.graspnet_dataset import convert_data_to_device

from sklearn.neighbors import KDTree

graspnet_k = None
graspnet_r = None


# def log_loss(epoch, i_batch, total_batch, loss, matches_loss, match_above_thresh_ratio, non_matches_loss, non_match_above_thresh_ratio, objectness_loss, writer, eval=False):
#     # print(f'Epoch:{epoch}, batch:{i_batch}\n-------loss:{loss.item()}\nmatches_loss:{matches_loss.item()}\nnon_matches_loss:{non_matches_loss.item()}\nnon_match_above_thresh_ratio:{non_match_above_thresh_ratio.item()}')
#     if eval:
#         print(f'\033[034mEpoch:{epoch}, batch:{i_batch}, total_batch:{total_batch}\n-------\033[0m\nloss:{loss}\nmatches_loss:{matches_loss}\nnon_matches_loss:{non_matches_loss}\nnon_match_above_thresh_ratio:{non_match_above_thresh_ratio}\nmatch_above_thresh_ratio:{match_above_thresh_ratio}\nobjectness_f1:{objectness_loss}')
#     else:
#         if writer is not None:
#             print(f'\033[034mEpoch:{epoch}, batch:{i_batch}, total_batch:{total_batch}\n-------\033[0m\nloss:{loss}\nmatches_loss:{matches_loss}\nnon_matches_loss:{non_matches_loss}\nnon_match_above_thresh_ratio:{non_match_above_thresh_ratio}\nmatch_above_thresh_ratio:{match_above_thresh_ratio}\nobjectness_f1:{objectness_loss}')
#             writer.add_scalar(f'loss/loss', loss, total_batch)
#             writer.add_scalar(f'loss/matches_loss', matches_loss, total_batch)
#             writer.add_scalar(f'loss/non_matches_loss', non_matches_loss, total_batch)
#             writer.add_scalar(f'loss/objectness_F1', objectness_loss, total_batch)
#             writer.add_scalar(f'loss/non_match_above_thresh_ratio', non_match_above_thresh_ratio, total_batch)
#             writer.add_scalar(f'loss/match_above_thresh_ratio', match_above_thresh_ratio, total_batch)


def log_loss(epoch, i_batch, total_batch, loss, match_loss, objectness_loss, match_f1, objectness_f1, writer,
             eval=False):
    # print(f'Epoch:{epoch}, batch:{i_batch}\n-------loss:{loss.item()}\nmatches_loss:{matches_loss.item()}\nnon_matches_loss:{non_matches_loss.item()}\nnon_match_above_thresh_ratio:{non_match_above_thresh_ratio.item()}')
    if eval:
        print(
            f'\033[034mEpoch:{epoch}, batch:{i_batch}, total_batch:{total_batch}\n-------\033[0m\nloss:{loss}\nmatches_loss:{match_loss}\nobjectness_loss:{objectness_loss}\nmatches_f1:{match_f1}\nobjectness_f1:{objectness_f1}')
    else:
        if writer is not None:
            print(
                f'\033[034mEpoch:{epoch}, batch:{i_batch}, total_batch:{total_batch}\n-------\033[0m\nloss:{loss}\nmatches_loss:{match_loss}\nobjectness_loss:{objectness_loss}\n\nmatches_f1:{match_f1}\nobjectness_f1:{objectness_f1}')
            writer.add_scalar(f'loss/loss', loss, total_batch)
            writer.add_scalar(f'loss/matches_loss', match_loss, total_batch)
            writer.add_scalar(f'loss/objectness_loss', objectness_loss, total_batch)
            writer.add_scalar(f'loss/matches_f1', match_f1, total_batch)
            writer.add_scalar(f'loss/objectness_f1', objectness_f1, total_batch)


def load_model_key_points(obj_id, return_colors=False):
    '''
    **Input:**
    
    - obj_id: int of the index of the object.

    - return_colors: bool of whether to return colors of points.

    **Output:**

    - numpy array of shape (KEY_POINT_NUM, 3)
    '''
    points = np.load(os.path.join(MODEL_KEY_POINT_DIR, '%03d_points.npy' % obj_id))
    keypoint_nums = np.load(os.path.join(MODEL_KEY_POINT_DIR, 'key_point_num.npy'))
    keypoint_num = keypoint_nums[obj_id]
    if not return_colors:
        points = points[0:keypoint_num][:]
        # print(points.shape)
        return points
    if return_colors:
        points = points[0:keypoint_num][:]
        colors = np.load(os.path.join(MODEL_KEY_POINT_DIR, '%03d_colors.npy' % obj_id))
        colors = colors[0:keypoint_num][:]
        # print(points.shape)
        return points, colors


def load_scene_model_key_points(scene_id, camera, ann_id, return_colors=False):
    '''
    **Input:**

    - scene_id: int of the scene index.

    - camera: string of the camera type.

    - ann_id: int of the annotation.

    - return_colors: bool of whether to return colors of points.

    **Output:**

    - np.array of shape (m, 1024, 3) of the model points in camera frame. m is the number of objects in the scene.
    '''

    graspnet_k = GraspNetEval(GRASPNET_ROOT, 'kinect', 'all')
    graspnet_r = GraspNetEval(GRASPNET_ROOT, 'realsense', 'all')

    if camera == 'kinect':
        graspnet = graspnet_k
    elif camera == 'realsense':
        graspnet = graspnet_r
    else:
        raise ValueError('camera must be either "kinect" or "realsense"')

    obj_list = []
    if return_colors:
        color_list = []
    obj_index_list, pose_list, _, _ = graspnet.get_model_poses(scene_id=scene_id, ann_id=ann_id)
    for i, obj_id in enumerate(obj_index_list):
        pose = np.asarray(pose_list[i])
        if return_colors:
            obj_key_points, obj_key_colors = load_model_key_points(obj_id, return_colors=True)
            if i == 0:
                color_list = obj_key_colors.tolist()
            else:
                color_list = color_list + obj_key_colors.tolist()
        else:
            obj_key_points = load_model_key_points(obj_id, return_colors=False)
        obj_key_points = np.column_stack((obj_key_points, np.ones((obj_key_points.shape[0], 1))))
        obj_in_scene_key_points = np.matmul(pose, obj_key_points.T).T
        obj_in_scene_key_points = np.delete(obj_in_scene_key_points, 3, axis=1)
        if i == 0:
            obj_list = obj_in_scene_key_points.tolist()
        else:
            obj_list = obj_list + obj_in_scene_key_points.tolist()
    if return_colors:
        return np.array(obj_list), np.array(color_list)
    else:
        return np.array(obj_list)


def load_label(scene_id, camera, ann_id):
    '''
    **Input:**

    - scene_id: int of the scene index.

    - camera: string of the camera type.

    - ann_id: int of the annotation.

    **Output:**

    - np.array of shape (720, 1280, 3) of the labels.
    '''
    label = np.load(os.path.join(LABEL_DIR, 'scene_%04d' % scene_id, camera, '%04d.npy' % ann_id))
    return label


def convert_data_to_device(dataloader_dict, device):
    '''
    **Input:**

    - dataloader_dict: dict of dataloader returned result.

    - device: string of device.

    **Output:**

    - dict of data.
    '''
    ret_dict = dict()
    for key in dataloader_dict.keys():
        if key in ['l', 'r']:
            lr_dict = dict()
            for lr_key in dataloader_dict[key].keys():
                lr_dict[lr_key] = dataloader_dict[key][lr_key].to(device)
            ret_dict[key] = lr_dict
        elif key != 'repr':
            ret_dict[key] = dataloader_dict[key].to(device)
        else:
            ret_dict[key] = dataloader_dict[key]
    return ret_dict


def get_key_point_dict(scene_id, ann_id, camera):
    '''
    - obj_id_list: a list of object ids
    :return: a mapping of id->beginning number
    '''
    graspnet_k = GraspNetEval(GRASPNET_ROOT, 'kinect', 'all')
    graspnet_r = GraspNetEval(GRASPNET_ROOT, 'realsense', 'all')
    if camera == 'kinect':
        graspnet = graspnet_k
    elif camera == 'realsense':
        graspnet = graspnet_r
    else:
        raise ValueError('camera must be either "kinect" or "realsense"')

    obj_id_list, _, _, _ = graspnet.get_model_poses(scene_id=scene_id, ann_id=ann_id)
    dict = {}
    # print(obj_id_list)
    keypoint_nums = np.load(os.path.join(MODEL_KEY_POINT_DIR, 'key_point_num.npy'))
    count = 0
    # print(obj_id_list)
    for i in range(len(obj_id_list)):
        obj_id = obj_id_list[i]
        dict[i] = count
        keypoint_num = keypoint_nums[obj_id]
        count = round(count + keypoint_num)
    # print(dict)
    return dict


def get_posed_points(points, pose):
    points = points.T  # (3, -1)
    points = np.matmul(pose[:3, :3], points)  # (3, -1)
    points = points.T + pose[:3, 3]  # (-1, 3)
    # points = np.column_stack((points,np.ones((points.shape[0],1))))
    # points = np.matmul(pose, points.T).T
    # points = np.delete(points, 3, axis=1)
    return points


def get_farthest_point(points_1, points_2):
    def norm(t):
        return np.sqrt(np.sum(t * t, axis=-1))
    points_1 = np.array(points_1)
    points_2 = np.array(points_2)
    points_1 = points_1[:, np.newaxis]
    points_2 = points_2[np.newaxis, :]
    dist = norm(points_1 - points_2)
    indices = np.argmax(dist, axis=-1)
    max_dist = dist[np.array(list(range(points_1.shape[0]))), indices]
    return max_dist, indices


def get_object_diagonal(points):
    xyz = np.array(points)
    return np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))


def get_diameter(points):
    # max_dist, indices = get_farthest_point(points, points)
    # return torch.max(max_dist)
    return get_object_diagonal(points)


def AUC(dist, threshold):
    dist.sort()
    N = len(dist)
    A = torch.tensor(0.0)
    cur_d = 0.0
    for idx, d in enumerate(dist):
        # print(d)
        cur_d = d
        if d > threshold:
            break
        if idx == 0:
            continue
        A += 0.5 * (2 * idx + 1) * (d - dist[idx - 1]) / (N * threshold)

    if cur_d <= threshold:
        A += (threshold - cur_d) / threshold
    return A


def load_scene_point_cloud(scene_id, camera, ann_id, down_sample_alpha=0.005):
    graspnet_k = GraspNetEval(GRASPNET_ROOT, 'kinect', 'all')
    graspnet_r = GraspNetEval(GRASPNET_ROOT, 'realsense', 'all')
    if camera == 'kinect':
        graspnet = graspnet_k
    elif camera == 'realsense':
        graspnet = graspnet_r
    else:
        raise ValueError('camera must be either "kinect" or "realsense"')
    points = graspnet.loadScenePointCloud(scene_id, camera, ann_id, use_mask=True)
    pcd = points.voxel_down_sample(down_sample_alpha)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def norm(t):
    return torch.sqrt(torch.sum(t * t, dim=-1))


def torch_get_closest_point(A, B, computing_device='cuda:0'):
    # print(f'calling with A:{A.shape},{A.dtype},{A.device}, B:{B.shape},{B.dtype},{A.device}')
    if A.shape[0] * B.shape[0] > 4e7:
        # assert False
        len_a = A.shape[0]
        width = int(1e7 / B.shape[0])
        iters = int((len_a - 1) / width) + 1
        # print('torch_get_closest_point case 1!!')
        # print(width,iters, len_a)
        min_dist = []
        indices = []
        for i in range(iters):
            A_row = A[i * width: min(len_a, (i + 1) * width)]
            i_dist, i_indices = torch_get_closest_point(A_row, B, computing_device=computing_device)
            min_dist += list(i_dist.detach().cpu().numpy())
            indices += list(i_indices.detach().cpu().numpy())
    #     A_row = A_row.unsqueeze(1)
    #     B = B.unsqueeze(0)
    #     dist = norm(A_row - B)
    #     # print('dist.shape:', dist.shape)
    # # print(f'A.shape:{A.shape}, B.shape:{B.shape}, dist.shape:{dist.shape}')
    #     indices = list(torch.argmin(dist, axis = -1))
    #     total_indices += indices
    #     print(total_indices)
    #     break
    #     min_dist_i = dist[torch.tensor(list(range(A_row.shape[0]))), indices]
    #     min_dist.append(min_dist_i)
    else:
        A = A.unsqueeze(1).to(computing_device)
        B = B.unsqueeze(0).to(computing_device)
        dist = norm(A - B)
        # print(f'A.shape:{A.shape}, B.shape:{B.shape}, dist.shape:{dist.shape}')
        indices = torch.argmin(dist, axis=-1)
        min_dist = dist[torch.tensor(list(range(A.shape[0]))), indices]
    return torch.tensor(min_dist), torch.tensor(indices)


def get_cloud(points, voxel_size=0.008):
    '''
    **Input:**
    - 
    '''
    points = torch.tensor(points).float()
    coords = (points / voxel_size).int()

    idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])
    sinput = ME.SparseTensor(points_batch, coords_batch)
    return sinput, idxs


def validation_on_test_dataset(net, loss_layer, test_dataloader, device,
                               total_batch=15 * 10 ** 5,
                               record_data_set=False,
                               record_name=None):
    if record_data_set:
        validation_set_repr = []

    with torch.no_grad():
        loss_collector = []
        match_f1_collector = []
        objectness_f1_collector = []
        print(64 * "=")
        print(24 * " " + 'VALIDATION')
        for i_batch_test, batch_data_test in tqdm.tqdm(enumerate(test_dataloader)):

            device_batch_data = convert_data_to_device(batch_data_test, device)
            if record_data_set:
                validation_set_repr += device_batch_data['repr']
            feat, obj_res = net(device_batch_data, total_batch=total_batch)
            loss, match_loss, objectness_loss, match_f1, objectness_f1 = loss_layer(feat, obj_res, verbose=False)
            loss_collector.append(loss.detach().cpu().numpy())
            match_f1_collector.append(match_f1)
            objectness_f1_collector.append(objectness_f1)

    if record_data_set:
        if record_name is None:
            record_name = 'validation_set_unknown'
        np.savetxt('./' + record_name + '.csv', np.array(validation_set_repr), fmt='%s')

    loss_collector = np.array(loss_collector).flatten()
    match_f1_collector = np.array(match_f1_collector).flatten()
    objectness_f1_collector = np.array(objectness_f1_collector).flatten()
    return loss_collector, match_f1_collector, objectness_f1_collector


def inference_batch_data(net, batch_data, device, total_batch=15 * 10 ** 5):
    with torch.no_grad():
        device_batch_data = convert_data_to_device(batch_data, device)
        feat, obj_res = net(device_batch_data, total_batch=total_batch)

        rgbxyz_l = batch_data['l']['points_batch'].detach().numpy().copy()
        rgbxyz_r = batch_data['r']['points_batch'].detach().numpy().copy()
        ori_idx_r = batch_data['r']['idxs'].detach().numpy().copy()
        sem_seg_label = batch_data['r']['sem_seg_label'].detach().numpy().copy()

        rgb_l = rgbxyz_l[:, 3:]
        points_l = rgbxyz_l[:, :3]
        rgb_r = rgbxyz_r[:, 3:]
        points_r = rgbxyz_r[:, :3]

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(points_r)
    pcd_r.colors = o3d.utility.Vector3dVector(rgb_r)

    sem_seg_pred = obj_res['sem_seg_pred'].reshape((-1, 2)).detach().cpu().numpy()
    sem_seg_idx = obj_res['idx'].detach().cpu().numpy().reshape(-1, )

    pos_idxs = sem_seg_idx[sem_seg_pred[:, 1] > sem_seg_pred[:, 0]]
    pos_idxs_gt = sem_seg_idx[sem_seg_label[sem_seg_idx] > 0.5]

    return pcd_r, pos_idxs, pos_idxs_gt
