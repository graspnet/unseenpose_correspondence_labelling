import open3d as o3d
import numpy as np
from .utils import load_label, load_scene_model_key_points, get_key_point_dict
from .utils import graspnet_k, graspnet_r
from .constant import IMAGE_HEIGHT, IMAGE_WIDTH, KEY_POINT_NUM
import os

# from annotations_correspondence import match_two_annotations

def draw_line(p1, p2, a = 9e-3, color = np.array((0.0,1.0,0.0))):
    '''
    **Input:**

    - p1: np.array of shape(3), the first point.

    - p2: np.array of shape(3), the second point.

    - a: float of the length of the square of the bottom face.

    **Output**

    - open3d.geometry.TriangleMesh of the line
    '''
    vertex_1 = o3d.geometry.TriangleMesh.create_box(1.5 * a,1.5 * a,1.5 * a)
    vertex_1.translate(p1 - np.array((0.75 * a, 0.75 * a, 0.75 * a)))
    vertex_1.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array((1.0,0,0)), (8, 1))) 
    vertex_2 = o3d.geometry.TriangleMesh.create_box(1.5 * a,1.5 * a,1.5 * a)
    vertex_2.translate(p2 - np.array((0.75 * a, 0.75 * a, 0.75 * a)))
    vertex_2.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array((1.0,0,0)), (8, 1))) 
    d = np.linalg.norm(p1 - p2)
    v1 = (p2 - p1) / d
    v2 = np.cross(np.array((0,0,1.0)), v1)
    v3 = np.cross(v1, v2)
    R = np.stack((v3, v2, v1)).astype(np.float64).T
    box = o3d.geometry.TriangleMesh.create_box(width = a, height = a, depth = d)
    box = box.translate(np.array((-a / 2, -a / 2, 0)))
    trans_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array((0,0,0,1))))
    # print('trans_matrix:{}'.format(trans_matrix))
    box = box.transform(trans_matrix)
    box = box.translate(p1)
    box.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1))) 
    return box, vertex_1, vertex_2

def vis_matches(data_dict):
    '''
    **Input:**

    - data_dict: dict returned by the dataloader.
    
    **Output:**

    - visualize matches or non-matches.
    '''
    matches = data_dict['matches']
    cloud_l = data_dict['l']['origin_cloud']
    cloud_r = data_dict['r']['origin_cloud']
    print(cloud_l.shape)

def vis_label(scene_id, camera, ann_id, if_show = True):
    '''
    **Input:**

    - scene_id: int of the scene index.

    - camera: string of the type of the camera:

    - ann_id: int of the annotaion index.

    **Output:**

    - no output but visualize the label.
    '''
    if camera == 'realsense':
        graspnet = graspnet_r
    elif camera == 'kinect':
        graspnet = graspnet_k
    label = load_label(scene_id = scene_id, camera = camera, ann_id = ann_id)
    scene_key_points, scene_key_colors = load_scene_model_key_points(scene_id = scene_id, camera = camera, ann_id = ann_id, return_colors = True)
    scene_points, scene_colors = graspnet.loadScenePointCloud(sceneId = scene_id, camera = camera, annId = ann_id, use_workspace = False, use_mask = False, format = 'numpy')
    print(f'scene_key_points.shape:{scene_key_points.shape}')
    scene_key_points[:,0] -= 1
    key_point_geometry = o3d.geometry.PointCloud()
    key_point_geometry.points = o3d.utility.Vector3dVector(scene_key_points)
    key_point_geometry.colors = o3d.utility.Vector3dVector(scene_key_colors)
    scene_geometry = o3d.geometry.PointCloud()
    scene_geometry.points = o3d.utility.Vector3dVector(scene_points.reshape((-1,3)))
    scene_geometry.colors = o3d.utility.Vector3dVector(scene_colors.reshape((-1,3)))
    
    print(f'scene points.shape:{scene_points.shape}')
    lines = []
    key_point_dict = get_key_point_dict(scene_id, ann_id, camera)

    for i in range(1000):
        if len(lines) > 20:
            break
        y, x = np.random.randint((IMAGE_HEIGHT)), np.random.randint((IMAGE_WIDTH))
        point_idx, obj_idx, dist = label[y,x]
        print(f'dist is {dist}')
        point_idx = round(point_idx)
        obj_idx = round(obj_idx)
    
        if dist < 0.01:
            p1 = scene_points[y,x]
            # key_point_index = int(key_point_dict[obj_idx]) + point_idx
            p2 = scene_key_points[int(obj_idx * KEY_POINT_NUM + point_idx)]
            lines.append(draw_line(p1, p2))
            print(f'y:{y}, x:{x}, point:{point_idx},obj:{obj_idx}, distance:{dist}')

    if if_show:
        o3d.visualization.draw_geometries([key_point_geometry, scene_geometry, *lines])
    else:
        pass

def vis_sampled_points(obj_idx, dir_root = ''):

    points = np.load(os.path.join(dir_root,'%03d_points.npy' %obj_idx))
    colors = np.load(os.path.join(dir_root,'%03d_colors.npy' %obj_idx))
    key_point_num = np.load(os.path.join(dir_root,'key_point_num.npy'))
    num = key_point_num[obj_idx]

    points = points[0:num][:]
    colors = colors[0:num][:]
    vis_key_point = o3d.geometry.PointCloud()
    vis_key_point.points = o3d.utility.Vector3dVector(points)
    vis_key_point.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([vis_key_point])


def vis_scene_ann_mapping(scene_id, camera,  ann_id_left, ann_id_right):
    if camera == 'realsense':
        graspnet = graspnet_r
    elif camera == 'kinect':
        graspnet = graspnet_k

    # left_key_points, right_key_points, label = match_two_annotations(scene_id, camera, ann_id_left, ann_id_right)

    left_key_points = np.load('ann_mapping/left.npy')
    right_key_points = np.load('ann_mapping/right.npy')
    label = np.load('ann_mapping/labels.npy')
    # left geometry
    scene_left_points, scene_left_colors = graspnet.loadScenePointCloud(sceneId=scene_id, camera=camera, annId=ann_id_left,
                                                              use_workspace=False, use_mask=False, format='numpy')
    left_geometry = o3d.geometry.PointCloud()
    left_geometry.points = o3d.utility.Vector3dVector(scene_left_points.reshape((-1,3)))
    left_geometry.colors = o3d.utility.Vector3dVector(scene_left_colors.reshape((-1,3)))

    # right geometry
    scene_right_points, scene_right_colors = graspnet.loadScenePointCloud(sceneId=scene_id, camera=camera,
                                                                        annId=ann_id_right,
                                                                        use_workspace=False, use_mask=False,
                                                                        format='numpy')
    right_geometry = o3d.geometry.PointCloud()
    scene_right_points = scene_right_points.reshape((-1, 3))
    scene_right_points[:,0] -= 1
    right_key_points[:,0] -= 1
    right_geometry.points = o3d.utility.Vector3dVector(scene_right_points)
    right_geometry.colors = o3d.utility.Vector3dVector(scene_right_colors.reshape((-1, 3)))

    lines = []
    for i in range(1000):
        if len(lines) > 20:
            break
        left_index = np.random.randint(label.shape[0])
        right_index, dist = label[left_index]
        right_index = round(right_index)

        if dist < 0.01:
            point_l = left_key_points[left_index]
            point_r = right_key_points[int(right_index)]
            lines.append(draw_line(point_l,point_r))

    o3d.visualization.draw_geometries([left_geometry,right_geometry,*lines])


def vis_graspnet_label(data, label, num_correspondence = 10):
    '''
    **Input:**

    - data: dict of the data pair.

    - label: torch.tensor of the label shape (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    
    **Output:**

    - No output but visualize the label.
    '''
    from .dataset.graspnet.graspnet_constant import RESIZE_WIDTH, RESIZE_HEIGHT
    
    cloud_l = data['l']['origin_cloud'].detach().numpy()
    cloud_r = data['r']['origin_cloud'].detach().numpy()
    cloud_r[:,1] = cloud_r[:,1] + 0.6
    cloud_r[:,2] = cloud_r[:,2] + 0.6
    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(cloud_l)
    pcd_l.colors = o3d.utility.Vector3dVector(data['l']['rgb'].detach().numpy().reshape((3,-1)).T)

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(cloud_r)
    pcd_r.colors = o3d.utility.Vector3dVector(data['r']['rgb'].detach().numpy().reshape((3,-1)).T)
    lines = []
    min_dist = label[:,:,2].reshape(-1)
    row_id = label[:,:,0].astype(int).reshape(-1)
    col_id = label[:,:,1].astype(int).reshape(-1)
    total_id = row_id * RESIZE_WIDTH + col_id
    for i in range(num_correspondence):
        rand_id = np.random.randint(RESIZE_HEIGHT * RESIZE_WIDTH)
        # print(f'rand_id:{rand_id}')
        while min_dist[rand_id] > 0.005:
            rand_id = np.random.randint(RESIZE_HEIGHT * RESIZE_WIDTH)
        # print(f'call with min_dist:{min_dist[rand_id]}')
        # print(f'l: id:{rand_id}, point:{cloud_l[rand_id]}, r: id:{total_id[rand_id]}, point:{cloud_r[total_id[rand_id]]}')
        l, v1, v2 = draw_line(cloud_l[rand_id], cloud_r[total_id[rand_id]])
        lines += [l, v1, v2]
    
    o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines])

def vis_graspnet_label_v2(data, num_correspondence = 10):
    '''
    **Input:**

    - data: dict of the data pair.
    
    **Output:**

    - No output but visualize the label.
    '''
    from .dataset.graspnet.graspnet_constant import RESIZE_WIDTH, RESIZE_HEIGHT
    label = data['labels'].detach().numpy()
    cloud_l = data['l']['cloud'].detach().numpy().copy()
    cloud_r = data['r']['cloud'].detach().numpy().copy()
    cloud_r[:,1] = cloud_r[:,1] + 0.6
    cloud_r[:,2] = cloud_r[:,2] + 0.6
    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(cloud_l)
    pcd_l.colors = o3d.utility.Vector3dVector(data['l']['rgb_selected'].detach().numpy())

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(cloud_r)
    pcd_r.colors = o3d.utility.Vector3dVector(data['r']['rgb_selected'].detach().numpy())
    lines = []
    min_dist = label[:, 1]
    total_id = label[:, 0].astype(int)
    num_point = len(min_dist)
    print(f'num points:{num_point}')
    print(f'left cloud:{cloud_l.shape}')
    print(f'right cloud:{cloud_r.shape}')
    for _ in range(num_correspondence):
        rand_id = np.random.randint(num_point)
        # print(f'rand_id:{rand_id}')
        while min_dist[rand_id] > 0.005:
            rand_id = np.random.randint(num_point)
        # print(f'call with min_dist:{min_dist[rand_id]}')
        # print(f'l: id:{rand_id}, point:{cloud_l[rand_id]}, r: id:{total_id[rand_id]}, point:{cloud_r[total_id[rand_id]]}')
        l, v1, v2 = draw_line(cloud_l[rand_id], cloud_r[total_id[rand_id]])
        lines += [l, v1, v2]
    
    o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines])

def vis_match_pairs(data, matches_key = 'matches', number = 10):
    '''
    **Input:**

    - data: dict of the data pair.

    - matches_key: string of 'matches' or 'non_matches'
    
    **Output:**

    - No output but visualize the matches or non_matches.

    '''
    from .dataset.graspnet.graspnet_constant import RESIZE_WIDTH, RESIZE_HEIGHT
    match = data[matches_key].detach().numpy().copy()
    np.random.shuffle(match)
    match_num = len(match)
    match = match[:min(number,match_num)]
    num_pair = match.shape[0]
    cloud_l = data['l']['origin_cloud'].detach().numpy().copy()
    cloud_r = data['r']['origin_cloud'].detach().numpy().copy()
    print(f'cloud_l.shape:{cloud_l.shape}')
    cloud_r[:,1] = cloud_r[:,1] + 0.6
    cloud_r[:,2] = cloud_r[:,2] + 0.6
    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(cloud_l)
    pcd_l.colors = o3d.utility.Vector3dVector(data['l']['rgb'].detach().numpy().reshape((3,-1)).T)

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(cloud_r)
    pcd_r.colors = o3d.utility.Vector3dVector(data['r']['rgb'].detach().numpy().reshape((3,-1)).T)
    lines = []
    for i in range(num_pair):
      l_id = match[i][0]
      r_id = match[i][1]
      l, v1, v2 = draw_line(cloud_l[l_id], cloud_r[r_id])
      lines += [l, v1, v2]
    o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines])

def vis_match_pairs_v2(data, matches_key = 'matches', number = 10):
    '''
    **Input:**

    - data: dict of the data pair.

    - matches_key: string of 'matches' or 'non_matches'
    
    **Output:**

    - No output but visualize the matches or non_matches.

    '''
    from .dataset.graspnet.graspnet_constant import RESIZE_WIDTH, RESIZE_HEIGHT
    match = data[matches_key].detach().numpy().copy()
    np.random.shuffle(match)
    match_num = len(match)
    match = match[:min(number,match_num)]
    num_pair = match.shape[0]
    cloud_l = data['l']['cloud'].detach().numpy().copy()
    cloud_r = data['r']['cloud'].detach().numpy().copy()
    print(f'cloud_l.shape:{cloud_l.shape}')
    cloud_r[:,1] = cloud_r[:,1] + 0.6
    cloud_r[:,2] = cloud_r[:,2] + 0.6
    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(cloud_l)
    pcd_l.colors = o3d.utility.Vector3dVector(data['l']['rgb_selected'].detach().numpy())

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(cloud_r)
    pcd_r.colors = o3d.utility.Vector3dVector(data['r']['rgb_selected'].detach().numpy())
    lines = []
    for i in range(num_pair):
      l_id = match[i][0]
      r_id = match[i][1]
      l, v1, v2 = draw_line(cloud_l[l_id], cloud_r[r_id])
      lines += [l, v1, v2]
    o3d.visualization.draw_geometries([pcd_l, pcd_r, *lines])
       
      

