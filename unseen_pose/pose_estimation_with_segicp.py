def crop_front_face(pcd):
    """
    Given a pcd (open3d.geometry.Pointcloud)
    Crop the point cloud front face from the visual direction
    """

    xyz = np.asarray(pcd.points)

    img_px_width = 640
    img_px_height = 320
    map_width = 2
    map_height = map_width * img_px_height / img_px_width

    dummy_value = 1234
    r = 1
    map_img = dummy_value * np.ones((img_px_height, img_px_width))


    for xyz_ in xyz:
        x_j = int(img_px_width * (xyz_[0] + map_width / 2) / map_width)
        y_i = int(img_px_height * (xyz_[1] + map_height / 2) / map_height)
        for j in range(x_j - r, x_j + r):
            for i in range(y_i - r, y_i + r):
                map_img[i, j] = min(map_img[i, j], xyz_[2])

    zmap = map_img
    xmap = np.arange(img_px_width)
    ymap = np.arange(img_px_height)
    xmap, ymap = np.meshgrid(xmap, ymap)

    xmap = xmap * map_width / img_px_width - map_width / 2
    ymap = ymap * map_height / img_px_height - map_height / 2

    xyz_front = np.stack([xmap, ymap, zmap], axis=-1).reshape((-1, 3))
    xyz_front = xyz_front[xyz_front[:, 2] != dummy_value]
    pcd_front = o3d.geometry.PointCloud()
    pcd_front.points = o3d.utility.Vector3dVector(xyz_front)
    return pcd_front


def evaluate_model_scene_alignment(pcd_o, pcd_sc, distance_thres=0.01):
    """
    evaluate the point clouds alignment,
    pcd_o: object point cloud
    pcd_sc: scene point cloud

    """
    xyz_obj = np.asarray(pcd_o.points)
    xyz_scene = np.asarray(pcd_sc.points)
    tree_scene = KDTree(xyz_scene)

    distance, indices_NN = tree_scene.query(xyz_obj, k=20, return_distance=True)
    # indices_NN = indices_NN[distance[:, 0] <= distance_thres].flatten()

    # count_indices = Counter(indices_NN)
    # unique_indices = filter(lambda i: count_indices[i] <= 1, count_indices)
    c = set([])
    for idx_cand, dist_cand in zip(indices_NN, distance):
        idx_cand = idx_cand[dist_cand < distance_thres]

        for idx_ in idx_cand:
            if idx_ not in c:
                c.add(idx_)
                break
    return len(c) / xyz_obj.shape[0]


def generate_pose_hypothesis(pcd_s, pcd_t, voxel_size=0.004, visualize=False):
    pcd_src = copy.deepcopy(pcd_s)
    pcd_tar = copy.deepcopy(pcd_t)

    pcd_src.remove_radius_outlier(nb_points=128, radius=0.05)
    pcd_tar.remove_radius_outlier(nb_points=128, radius=0.05)

    pcd_src = pcd_src.voxel_down_sample(voxel_size)
    pcd_tar = pcd_tar.voxel_down_sample(voxel_size)

    t_src = pcd_src.get_center().reshape(3, 1)
    t_tar = pcd_tar.get_center().reshape(3, 1)

    T_init = np.identity(4)

    pose_hypothesis_collection = []
    T_optimal = T_init
    score_optimal = 0

    for i in range(5):
        axe = np.random.rand(3)
        axe = axe / np.linalg.norm(axe)
        for k in range(1, 6):
            angle = k * 1 / 6 * 2 * np.pi
            R_init = axangle2mat(axe, angle)
            T_init[:3, :3] = R_init
            t_init = t_tar - R_init@t_src
            T_init[:3, [3]] = t_init

            T_ = T_init
            pcd_transformed = copy.deepcopy(pcd_src)
            pcd_transformed.transform(T_init)

            T = icp(np.asarray(pcd_transformed.points), np.asarray(pcd_tar.points), max_iter=5)
            pcd_transformed.transform(T)
            T_ = T@T_

            pcd_cropped = crop_front_face(pcd_transformed)

            # l is the object point cloud;  r is the scene point cloud
            #  BLUE                                RED
            score = evaluate_model_scene_alignment(pcd_cropped, pcd_tar, distance_thres=voxel_size*1.1)
            print(score)

            pcd_cropped.paint_uniform_color([0, 0.651, 0.929])
            pcd_tar.paint_uniform_color([0.929, 0, 0.351])
            # o3d.visualization.draw_geometries([pcd_cropped, pcd_tar])

            T = icp(np.asarray(pcd_cropped.points), np.asarray(pcd_tar.points), max_iter=40)
            T_ = T@T_
            pcd_cropped.transform(T)
            pcd_transformed.transform(T)

            score = evaluate_model_scene_alignment(pcd_cropped, pcd_tar, distance_thres=voxel_size*1.1)
            print(score)
            print(32 * '-=')
            pcd_cropped.translate([0, 0, 0.005])
            # if score > 0.5:
            #     o3d.visualization.draw_geometries([pcd_cropped, pcd_tar])

            if score > score_optimal:
                T_optimal = T_
                score_optimal = score
    return T_optimal, score_optimal
