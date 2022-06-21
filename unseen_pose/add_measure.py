from .utils import get_posed_points, get_diameter, AUC
import torch
import numpy as np


def ADD(cloud, est_pose, real_pose):
    est_cloud = get_posed_points(cloud, est_pose) #(n,3)
    real_cloud = get_posed_points(cloud, real_pose) #(n,3)
    est_cloud = torch.from_numpy(est_cloud)
    real_cloud = torch.from_numpy(real_cloud)
    dist = torch.sqrt(torch.sum(torch.square(real_cloud - est_cloud),axis=1))
    return dist.cpu().numpy().mean()


def ADD_PCD(cloud_est, cloud_real):
    cloud_est = torch.from_numpy(cloud_est)
    cloud_real = torch.from_numpy(cloud_real)
    return torch.sqrt(torch.sum(torch.square(cloud_real - cloud_est),axis=1)).mean()


def ADD_AUC(cloud, est_pose, real_pose, threshold=None):

    if threshold is None:
        threshold = 0.5 * get_diameter(cloud)

    dist = ADD(cloud, est_pose, real_pose)
    dist = dist.sort()
    return AUC(dist[0], threshold)


def ADD_PCD_AUC(cloud_est, cloud_real, threshold=None):
    if threshold is None:
        threshold = 0.5 * get_diameter(cloud_est)
    dist = ADD_PCD(cloud_est, cloud_real)
    dist = dist.sort()
    return AUC(dist[0], threshold)
