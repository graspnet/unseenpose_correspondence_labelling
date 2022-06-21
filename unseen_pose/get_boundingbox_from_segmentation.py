import numpy as np
import cv2
# import MinkowskiEngine as ME
import argparse
import time
import os
import open3d as o3d

from unseen_pose.dataset.graspnet.graspnet_constant import HEATMAP_WIDTH, ORIGIN_WIDTH, ORIGIN_HEIGHT


def get_heatmap_from_point_cloud(points,
                                 mean_xyz,
                                 heatmap_width=HEATMAP_WIDTH,
                                 img_px_width=ORIGIN_WIDTH,
                                 img_px_height=ORIGIN_HEIGHT,
                                 sigma=10,
                                 r=10):
    """
    points: xyz coordinates of the point cloud segmented
    sigma: float: the standard deviation of the gaussian kernel used to produce the heat map
    r: the ratio of influence of the point
    return:
    mask: np.array of shape img_px_height * img_px_width
    """
    points_xy = points[:, :2]
    mean_xy = mean_xyz[:2]

    heatmap_height = heatmap_width * img_px_height / img_px_width
    heat_map_img = np.zeros((img_px_height, img_px_width))

    for xy_ in points_xy:
        x_j = int(img_px_width * (xy_[0] + heatmap_width / 2 - mean_xy[0]) / heatmap_width)
        y_i = int(img_px_height * (xy_[1] + heatmap_height / 2 - mean_xy[1]) / heatmap_height)
        for j in range(x_j - r, x_j + r):
            for i in range(y_i - r, y_i + r):
                i = max(min(i, img_px_height-1), 0)
                j = max(min(j, img_px_width-1), 0)
                heat_map_img[i, j] += np.exp(-((x_j - j) ** 2 + (y_i - i) ** 2) / sigma ** 2)
    return heat_map_img


def get_contour_from_heat_map(heat_map):
    """
    heat_map: 2d np.array of float

    """
    # convert the heat map to a gray image
    gray = (255 * (heat_map / np.max(heat_map))).astype(np.uint8)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    box_list = np.empty((len(cnts), 4))
    for i_, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        box_list[i_, :] = np.array([x, y, w, h])
    return thresh, box_list


def get_contoured_points_from_bounding_box(points_all, mean_xyz, x, y, w, h,
                                           heatmap_height, heatmap_width, img_px_height, img_px_width):
    x_bl = x * heatmap_width / img_px_width - heatmap_width / 2 + mean_xyz[0]
    x_br = (x + w) * heatmap_width / img_px_width - heatmap_width / 2 + mean_xyz[0]
    y_bl = y * heatmap_height / img_px_height - heatmap_height / 2 + mean_xyz[1]
    y_br = (y + h) * heatmap_height / img_px_height - heatmap_height / 2 + mean_xyz[1]
    contour_mask = (points_all[:, 0] > x_bl) * \
                   (points_all[:, 0] < x_br) * \
                   (points_all[:, 1] > y_bl) * \
                   (points_all[:, 1] < y_br) > 0
    points_contoured = points_all[contour_mask]
    return points_contoured, contour_mask


def compute_bbox_intersection(box_1, box_2):
    lower_bounds = np.maximum(box_1[:2], box_2[:2])  # (2,)
    upper_bounds = np.minimum(box_1[:2] + box_1[2:], box_2[:2] + box_2[2:])  # (2,)
    intersection_box = upper_bounds - lower_bounds
    if np.any(intersection_box < 0):
        return 0
    return intersection_box[0] * intersection_box[1]


def compute_bbox_distance(box_1, box_2):
    lower_bounds = np.maximum(box_1[:2], box_2[:2])  # (2,)
    upper_bounds = np.minimum(box_1[:2] + box_1[2:], box_2[:2] + box_2[2:])
    intersection_box = upper_bounds - lower_bounds
    if np.all(intersection_box > 0):
        return 0
    return -min(intersection_box)


def merge_bbox(box_1, box_2):
    lower_bounds = np.minimum(box_1[:2], box_2[:2])  # (2,)
    upper_bounds = np.maximum(box_1[:2] + box_1[2:], box_2[:2] + box_2[2:])
    merged_box = np.zeros(4)
    merged_box[:2] = lower_bounds
    merged_box[2:] = upper_bounds - lower_bounds
    return merged_box


def merge_and_rearrange_bbox(bbox_list, area_thres=100, distance_thres=10):

    bbox_list = bbox_list[bbox_list[:, 3] * bbox_list[:, 2] > area_thres]  # filter the box with area < threshold
    if len(bbox_list) >= 2:
        i = 0
        while i < len(bbox_list):
            j = i + 1
            while j < len(bbox_list):
                # print(bbox_list[i, :], bbox_list[j, :], compute_bbox_distance(bbox_list[i, :], bbox_list[j, :]))
                if compute_bbox_distance(bbox_list[i, :], bbox_list[j, :]) < distance_thres:
                    bbox_list[i] = merge_bbox(bbox_list[i, :], bbox_list[j, :])
                    # print(bbox_list[i, :], bbox_list[j, :])
                    bbox_list = np.delete(bbox_list, j, 0)
                else:
                    j += 1
            i += 1

    return bbox_list


def get_points_in_box_from_point_cloud(points_all,
                                       points_segmented,
                                       heatmap_width=HEATMAP_WIDTH,
                                       img_px_width=ORIGIN_WIDTH,
                                       img_px_height=ORIGIN_HEIGHT,
                                       sigma=8,
                                       r=10,
                                       margin=0,
                                       heat_threshold=10):
    heatmap_height = heatmap_width * img_px_height / img_px_width
    mean_xyz = points_segmented.mean(axis=0)
    heatmap = get_heatmap_from_point_cloud(points_segmented,
                                           mean_xyz,
                                           heatmap_width=heatmap_width,
                                           img_px_width=img_px_width,
                                           img_px_height=img_px_height,
                                           sigma=sigma,
                                           r=r)
    boxlist = get_contour_from_heat_map(heatmap).astype(np.int32)
    # boxlist = merge_and_rearrange_bbox(boxlist)

    res = []
    for c in boxlist:
        # x, y, w, h = c[0] - margin, c[1] - margin, c[2] + 2 * margin, c[3] + 2 * margin
        x, y, w, h = c[0] - margin, c[1] - margin, c[2] + 2 * margin, c[3] + 2 * margin

        points_contoured, contour_mask = get_contoured_points_from_bounding_box(points_all,
                                                                                mean_xyz,
                                                                                x, y, w, h,
                                                                                heatmap_height,
                                                                                heatmap_width,
                                                                                img_px_height,
                                                                                img_px_width)
        # print(points_contoured)
        xy = points_all[:, :2]

        heatmap_wh = np.array([heatmap_width, heatmap_height])
        img_px_wh = np.array([img_px_width, img_px_height])
        xy_in_img = np.floor((xy + heatmap_wh / 2 - mean_xyz[:2]) / heatmap_wh * img_px_wh).astype(np.int32)
        heat_value = heatmap[xy_in_img[:, 1].clip(0, img_px_height-1), xy_in_img[:, 0].clip(0, img_px_width-1)]
        heat_mask = heat_value > heat_threshold
        mask = np.logical_and(contour_mask, heat_mask)

        points_contoured = points_all[mask]

        if len(points_contoured) > 0:
            res.append((points_contoured, mask))
    return res, heatmap
