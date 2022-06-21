import copy

import numpy as np
import math
import open3d as o3d
import os
import torch


def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert (R_est.shape == R_gt.shape == (3, 3))
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def te(t_est, t_gt):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    assert (t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error


def get_object_diagonal(pcd):
    xyz = np.asarray(pcd.points)
    return 0.1 * np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))


def ADD(pcd, T_est, T_gt, threshold=None):
    if threshold is None:
        threshold = get_object_diagonal(pcd)

    pcd_gt = copy.deepcopy(pcd)
    pcd_est = copy.deepcopy(pcd)

    pcd_gt.transform(T_gt)
    pcd_est.transform(T_est)
    xyz_gt = np.asarray(pcd_gt.points)
    xyz_est = np.asarray(pcd_est.points)
    add = (np.sqrt((xyz_gt - xyz_est) ** 2)).sum(axis=1).mean()

    return add, add < threshold


def calc_pose_error(T_est, T_gt):
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3].reshape(-1, 1)
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3].reshape(-1, 1)
    return {
        'R error': re(R_est, R_gt),
        't error': te(t_est, t_gt)
    }
