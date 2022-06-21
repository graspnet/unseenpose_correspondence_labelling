import copy

import numpy as np
from itertools import product
import json
from .add_measure import ADD, ADD_PCD
from .adds_measure import ADDS
from .utils import get_posed_points


class RotationalAxis():
    def __init__(self, axis, rotation_num):
        """define the rotation matrix.

        Args:
            axis(np.array(3)): The rotation axis.
            rotation_num(int): The number of rotations. np.inf for infinity.
        """
        self.axis = axis
        self.rotation_num = rotation_num
    
    def get_rotation_matrix(self, theta):
        """Calculate the rotational matrix around the axis
        
        Args:
            theta(float): Angle in arc.

        Returns:
            np.array(3, 3): Rotation matrix.
        """
        return np.cos(theta) * np.eye(3) \
            + (1 - np.cos(theta)) * np.matmul(self.axis.reshape((3, 1)), self.axis.reshape((1, 3))) \
            + np.sin(theta) * np.array([
                [0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0]
            ])

    def iter_all_rotation_matrix(self, sample_number = 30):
        num_angles = self.rotation_num if not self.rotation_num == np.inf else sample_number
        step = np.pi / num_angles * 2.0
        cur_angle = 0
        for i in range(num_angles):
            cur_angle += step
            yield self.get_rotation_matrix(cur_angle)


class ObjectRotationalAxes():
    def __init__(self, axes_type = "no", sample_number = 30):
        self.axes_type = axes_type # (no, finite, infinite)
        if not self.axes_type in ["no", "finite", "infinite"]:
            raise ValueError("axes_type must be 'no', 'finite' or 'infinite'")
        self.sample_number = sample_number
        self.axes_list = []

    def add_axis(self, axis):
        self.axes_list.append(axis)

    def iter_all_poses(self):
        """get all poses if finite axes
        Returns:
            list: list of np.array(4, 4) poses.
        """
        if self.axes_type == "infinite":
            raise ValueError("cannot calculate all poses for 'infinite' axes object")
        elif self.axes_type == "no":
            yield np.eye(4, dtype=np.float32)
        else: # "finite"
            for pose_list in product(*[list(axis.iter_all_rotation_matrix()) for axis in self.axes_list]):
                if len(pose_list) == 1:
                    rotation_matrix = pose_list[0]
                else:
                    rotation_matrix = np.linalg.multi_dot(pose_list)
                yield np.vstack((
                    np.hstack((
                        rotation_matrix,
                        np.array([0, 0, 0.0]).reshape((3, 1))
                    )),
                    np.array([0, 0, 0, 1.0])
                ))

    def load_from_xyzdict(self, xyzdict):
        if xyzdict["x"] == np.inf and xyzdict["y"] == np.inf and xyzdict["z"] == np.inf:
            self.axes_type = "infinite"
            self.axes_list = []
        elif xyzdict["x"] == 1 and xyzdict["y"] == 1 and xyzdict["z"] == 1:
            self.axes_type = "no"
            self.axes_list = []
        else:
            self.axes_type = "finite"
            self.axes_list = []
            axis_dict = []
            for key_id, key in enumerate(["x", "y", "z"]):
                if xyzdict[key] > 1:
                    axis = np.zeros(3, dtype = np.float32)
                    axis[key_id] = 1.0
                    self.add_axis(RotationalAxis(axis, xyzdict[key]))


def IADD(cloud, est_pose, real_pose, obj_rotational_axes):
    """Implementation of the Infimum ADD(IADD).

    Args:
        cloud(np.array(n, 3)): The object point cloud.
        est_pose(np.array(n, 4)): The estimated pose.
        real_pose(np.array(n, 4)): The ground truth pose.
        obj_property(ObjectRotationalAxes): Rotational axes of the object.

    Returns:
        float: the IADD score.
    """
    if obj_rotational_axes.axes_type == "no":
        return ADD(cloud, est_pose, real_pose)
    elif obj_rotational_axes.axes_type == "finite":
        return min([ADD(cloud, est_pose, np.matmul(real_pose, one_pose)) for one_pose in obj_rotational_axes.iter_all_poses()])
    else: # "infinite"
        return np.linalg.norm(est_pose[:3, 3] - real_pose[:3, 3])


def IADD_PCD(cloud_est, cloud_real):
    if obj_rotational_axes.axes_type == "no":
        return ADD_PCD(cloud_est, cloud_real)
    elif obj_rotational_axes.axes_type == "finite":
        cloud_real_copy = copy.deepcopy(cloud_real)
        min_dist = ADD_PCD(cloud_est, cloud_real)
        for one_pose in obj_rotational_axes.get_all_poses():
            cloud_real_copy = get_posed_points(cloud_real_copy, one_pose)
            min_dist = min(min_dist, ADD_PCD(cloud_est, cloud_real_copy))
        return min_dist
    else:  # "infinite"
        return np.linalg.norm(cloud_est.mean(axis=0) - cloud_real.mean(axis=0))


def IADD_AUC(cloud, est_pose, real_pose, obj_rotational_axes, threshold=None):
    if threshold is None:
        threshold = 0.5 * get_diameter(cloud)
    dist = IADD(cloud, est_pose, real_pose, obj_rotational_axes)
    dist = dist.sort()
    return AUC(dist[0], threshold)


def load_json_annotation(json_path):
    with open(json_path) as json_file:
        anno = json.load(json_file)
    for obj_name in anno:
        for key in ["x", "y", "z"]:
            if anno[obj_name][key] == "inf":
                anno[obj_name][key] = np.inf
    return anno


if __name__ == "__main__":
    # r = RotationalAxis(np.array([0,0,1.0]), 4)
    # print(r.get_rotation_matrix(np.pi / 2))
    # for rr in r.iter_all_rotation_matrix():
    #     print(rr)

    # xyzdict = {"x" : 1, "y" : np.inf, "z" : 2}
    # axes = ObjectRotationalAxes()
    # axes.load_from_xyzdict(xyzdict)
    # for id, p in enumerate(axes.iter_all_poses()):
    #     print("pose {}:\n{}".format(id, p))

    json_path = "test.json"
    anno = load_json_annotation(json_path = json_path)
    for obj_name in anno.keys():
        print("Obj: {}".format(obj_name))
        xyzdict = anno[obj_name]
        axes = ObjectRotationalAxes()
        axes.load_from_xyzdict(xyzdict)
        if axes.axes_type == "infinite":
            print("infinite")
        else:
            for id, p in enumerate(axes.iter_all_poses()):
                print("pose {}:\n{}".format(id, p))
