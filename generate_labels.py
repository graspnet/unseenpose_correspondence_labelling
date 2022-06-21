import torch
import numpy as np
import torch.nn as nn
from multiprocessing import Pool
from unseen_pose.dataset.graspnet.graspnet_dataset import GraspNetDataset
from unseen_pose.constant import GRASPNET_ROOT, GRASPNET_ROOT_REAL, LABEL_DIR, LABEL_DIR_REAL
import sys
from tqdm import tqdm


def generate_label(cur_id, total, isreal=False):
    if isreal:
        root_path = GRASPNET_ROOT_REAL
        label_path = LABEL_DIR_REAL
        list_file_dir = 'graspnet_offline_list_real'
        camera = 'realsense'
    else:
        root_path = GRASPNET_ROOT
        label_path = LABEL_DIR
        list_file_dir = 'graspnet_offline_list'
        camera = 'blender_proc'

    dataset = GraspNetDataset(root=root_path, camera=camera, split='test',
                              list_file_dir=list_file_dir, matching_devices='cuda:{}'.format(cur_id % 6),
                              label_dir=label_path, generate_label=False, use_remove_mask=False, scene_pair_ratio=0.0,
                              model_pair_ratio=1.0, use_augmentation=False)
    step = len(dataset) // total
    end_range = (cur_id + 1) * step
    if cur_id == total - 1:
        end_range = len(dataset)
    for index in tqdm(range(cur_id * step, end_range), 'generating labels'):
        dataset.__getitem__(index)


if __name__ == '__main__':
    cur_id = int(sys.argv[1])
    total = int(sys.argv[2])
    isreal = int(sys.argv[3]) > 0.5
    generate_label(cur_id, total, isreal)
