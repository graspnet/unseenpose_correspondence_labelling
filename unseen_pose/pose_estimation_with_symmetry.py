import copy
from transforms3d.axangles import axangle2mat

from sklearn.neighbors import KDTree
import numpy as np
import open3d as o3d
import os
import cv2
from collections import Counter

INFERENCE_DIR = './inference_result'


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=1)


def get_closest_to_mean(xyz):
    """
    xyz: point cloud with coordinates shape of N * 3
    get the index of the closest point to the barycenter of the xyz
    """

    assert len(xyz.shape) == 2 and xyz.shape[0] > 0, "need input with two dimension"
    if xyz.shape[0] == 1:
        return 0

    xyz_mean = xyz.mean(axis=0)
    return np.argmin(((xyz - xyz_mean) ** 2).sum(axis=1))


def sample_farthest_points(xyz, k, initial_idx=None, metrics=l2_norm,
                           skip_initial=False, indices_dtype=np.int32,
                           distances_dtype=np.float32):
    """
    Apply the Farthest Point Sampling on a point cloud with coordinates (xyz).
    The number of points sampled is: k

    metrics: the function used to decide the distance between two points
    skip_initial: boolean value, if skip the random initial
    initial_idx: integer, the initial index used to start the FPS, if None, random pick one

    return: indices: the indices of the sampled points in original point cloud
            distances: the distance from each points in point cloud to the points sampled
    """

    num_point, coord_dim = xyz.shape
    indices = np.zeros(k, dtype=indices_dtype)
    distances = np.zeros((k, num_point), dtype=distances_dtype)

    if initial_idx is None:
        indices[0] = np.random.randint(len(xyz))
    else:
        indices[0] = initial_idx

    farthest_point = xyz[indices[0]]

    min_distances = metrics(farthest_point[None, :], xyz)

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[0] = np.argmax(min_distances)
        farthest_point = xyz[indices[0]]
        min_distances = metrics(farthest_point[None, :], xyz)
    distances[0, :] = min_distances

    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        farthest_point = xyz[indices[i]]
        dist = metrics(farthest_point[None, :], xyz)
        distances[i, :] = dist
        min_distances = np.minimum(min_distances, dist)

    return indices, distances


def generate_fragments(xyz_all, k):
    """
    divide the point cloud (with xyz_all as coordinates) into k fragments
    the centers of the fragments are sampled using Farthest Point Sample
    xyz_all: np.array with shape N * 3
    k: int
    return the fragment indices of the point cloud: np.array of shape (N, )
    """
    indices_centers, _ = sample_farthest_points(xyz_all, k=k, skip_initial=True)
    xyz_centers = xyz_all[indices_centers]
    tree = KDTree(xyz_centers)
    indices_NN = tree.query(xyz_all, k=1, return_distance=False)
    return indices_NN.flatten()


def generate_correspondence_candidates_quartet(i_keypoint,
                                               xyz_candidates,
                                               indices_fragment_candidates,
                                               confidence_candidates):
    res = []
    for i_f in np.unique(indices_fragment_candidates):
        idx_f = np.arange(0, xyz_candidates.shape[0])[indices_fragment_candidates == i_f]
        xyz_fragment = xyz_candidates[idx_f]
        confidence_fragment = confidence_candidates[idx_f]
        candidate_delegate = idx_f[get_closest_to_mean(xyz_fragment)]
        res.append([candidate_delegate, i_keypoint, i_f, confidence_fragment.sum()/confidence_candidates.sum()])

    return res
