from .utils import get_posed_points, AUC, get_diameter
import torch
import numpy as np


def ADDS(cloud, est_pose, real_pose):
    est_cloud = get_posed_points(cloud, est_pose)  # (n,3)
    real_cloud = get_posed_points(cloud, real_pose)  # (n,3)
    return compute_adds(est_cloud, real_cloud).mean()


def get_closest_point(points_1, points_2):
    def norm(t):
        return np.sqrt(np.sum(t * t, axis=-1))
    points_1 = np.array(points_1)
    points_2 = np.array(points_2)
    points_1 = points_1[:, np.newaxis]
    points_2 = points_2[np.newaxis, :]
    dist = norm(points_1 - points_2)
    indices = np.argmin(dist, axis=-1)
    min_dist = dist[np.array(list(range(points_1.shape[0]))), indices]
    return min_dist, indices


def compute_adds(points_1, points_2):
    # each of point in points_1 to best point in points_2
    min_dist, indices = get_closest_point(points_1, points_2)
    return min_dist


def ADDS_PCD(cloud_est, cloud_real):
    return compute_adds(cloud_est, cloud_real).mean()


def ADDS_AUC(cloud, est_pose, real_pose, threshold=None):

    if threshold is None:
        threshold = 0.5 * get_diameter(cloud)

    dist = ADDS(cloud, est_pose, real_pose)
    dist.sort()
    return AUC(dist, threshold)


def ADDS_PCD_AUC(cloud_est, cloud_real, threshold=None):
    if threshold is None:
        threshold = 0.5 * get_diameter(cloud_est)
    dist = ADDS_PCD(cloud_est, cloud_real)
    dist.sort()
    return AUC(dist, threshold)
