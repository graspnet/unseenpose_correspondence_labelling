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
from unseen_pose.dataset.graspnet.graspnet_constant import PAIR_NUM, SCENE_PAIR_NUM, ANN_NUM, RESIZE_WIDTH, \
    RESIZE_HEIGHT, ORIGIN_HEIGHT, ORIGIN_WIDTH, CLOSE_DIST_THRES, REMOTE_DIST_THRES
from unseen_pose.constant import VOXEL_SIZE
from graspnetAPI.utils.utils import create_point_cloud_from_depth_image, CameraInfo, parse_posevector


def load_rgbd_pointcloud(rgb_path, depth_path, intrinsic, factor_depth, camera='blender_proc'):

    if camera == 'ycbv':
        ORIGIN_WIDTH = 640
        ORIGIN_HEIGHT = 480
        RESIZE_WIDTH = 480
        RESIZE_HEIGHT = 360
    elif camera == 'tless':
        ORIGIN_WIDTH = 720
        ORIGIN_HEIGHT = 540
        RESIZE_WIDTH = 480
        RESIZE_HEIGHT = 360
    else:
        ORIGIN_WIDTH = 1280
        ORIGIN_HEIGHT = 720
        RESIZE_WIDTH = 384
        RESIZE_HEIGHT = 288

    resize = transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH))
    totensor = transforms.ToTensor()
    trans_list = [resize, totensor]
    rgb_transform = transforms.Compose(trans_list)
    rgb = Image.open(rgb_path)
    rgb = rgb_transform(rgb)

    depth = Image.open(depth_path)
    # depth_np = np.load(depth_path)
    # depth_np[depth_np > 2 * factor_depth] = 0
    # depth = Image.fromarray(depth_np)
    depth = np.array(depth.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.NEAREST))
    sx = RESIZE_WIDTH / ORIGIN_WIDTH
    sy = RESIZE_HEIGHT / ORIGIN_HEIGHT
    camera = CameraInfo(RESIZE_WIDTH, RESIZE_HEIGHT, sx * intrinsic[0][0], sy * intrinsic[1][1],
                        sx * intrinsic[0][2] + 0.5 * sx - 0.5, sy * intrinsic[1][2] + 0.5 * sy - 0.5,
                        factor_depth)

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud = torch.tensor(np.transpose(cloud, (2, 0, 1))).float()
    return rgb, cloud


def load_model_pointcloud(pcd_path, mm2m=False):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(0.001)
    xyz = np.asarray(pcd.points)
    if mm2m:
        xyz = 0.001 * xyz
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = torch.from_numpy(np.asarray(pcd.colors))  # (n, 3)
    cloud = torch.from_numpy(np.asarray(pcd.points))
    return rgb, cloud


def graspnet_collate_fn(batch):
    ret_dict = dict()
    for i, b in enumerate(batch):
        if b['matches'] == None:
            b['matches'] = torch.tensor([[0, 0]])
        if b['non_matches'] == None:
            b['non_matches'] = torch.tensor([[0, 0]])

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

    ret_dict['matches'] = torch.stack(
        [batch[i]['matches'] for i in range(len(batch))])
    ret_dict['non_matches'] = torch.stack(
        [batch[i]['non_matches'] for i in range(len(batch))])
    ret_dict['l2r_pose'] = torch.stack(
        [batch[i]['l2r_pose'] for i in range(len(batch))])
    return ret_dict


def load_inference_dataset(pcd_l, pcd_r, voxel_size=VOXEL_SIZE):

    ret_dict = {}

    for lr_id, lr in enumerate(['l', 'r']):

        if lr == 'l':
            cloud = torch.from_numpy(np.asarray(pcd_l.points))
            rgb = torch.from_numpy(np.asarray(pcd_l.colors))
        else:
            cloud = torch.from_numpy(np.asarray(pcd_r.points))
            rgb = torch.from_numpy(np.asarray(pcd_r.colors))

        origin_cloud = cloud
        mask = None
        objectness_mask = torch.ones((cloud.size()[0],), dtype=torch.int32)
        sem_seg_mask = torch.ones((cloud.size()[0],), dtype=torch.int32)

        coords = torch.tensor(np.ascontiguousarray(cloud / voxel_size, dtype=np.float32))
        # print(coords)
        _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[idxs]
        cloud = cloud[idxs].float()  # (n, 3)
        objectness_label = objectness_mask[idxs]
        sem_seg_label = sem_seg_mask[idxs]

        if lr == 'r':
            rgb_selected = rgb[idxs]
        else:
            rgb_selected = rgb[idxs]

        mask_ = objectness_mask
        ret_dict[lr] = {'rgb': rgb, 'idxs': idxs, 'coords': coords, 'rgb_selected': rgb_selected,
                        'cloud': cloud, 'origin_cloud': origin_cloud, 'mask': mask_,
                        'objectness_label': objectness_label, 'sem_seg_label': sem_seg_label}

    l2r_pose = torch.eye(4)
    ret_dict['labels'] = torch.tensor([[0, 0.0]])
    ret_dict['matches'] = torch.tensor([[0, 0]])
    ret_dict['non_matches'] = torch.tensor([[0, 0]])
    ret_dict['l2r_pose'] = l2r_pose
    batch = [ret_dict]

    return graspnet_collate_fn(batch)
