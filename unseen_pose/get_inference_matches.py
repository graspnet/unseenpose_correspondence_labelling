import torch
import torch.nn as nn
import numpy as np
from .utils import torch_get_closest_point


def get_closest_point(l_feature, r_feature):
    '''
    **Input:**

    - l_feature: (l_points, 256).

    - r_feature: (r_points, 256).

    **Output:**

    - dist, indices (l_points), (l_points)
    '''
    norm_l_feature = nn.functional.normalize(l_feature, dim=1)
    norm_r_feature = nn.functional.normalize(r_feature, dim=1)
    dists = []
    indices = []
    for l_f in norm_l_feature:
        # print(f'l_f:{l_f}')
        lf_tensor = l_f.unsqueeze(0)
        dist, indice = torch_get_closest_point(lf_tensor, norm_r_feature)
        # print(f'dist:{dist}, indice:{indice}')
        dists.append(dist)
        indices.append(indice)
    return torch.tensor(dists, device=norm_l_feature.device), torch.tensor(indices, device=norm_l_feature.device)


def get_inference_matches(feat,
                          objectness_mask=None,
                          method='random',
                          init_dist_thresh=0.001,
                          iter_factor=1.25,
                          match_number=30):
    '''
    **Input:**

    - feat:
    'feature': (num_points, 256)
    
    - method: string of the sampling method, 'random' for randomly sampling, 'top' for top.

    - init_dist_thresh: valid for 'top', which is the first dist thresh.

    - iter_factor: valid for 'top', which is the factor that multiplied by the dist thresh.

    - match_number: valid for both method, how many number matches to return.

    **Output:**
    
    - torch.tensor of device cpu of shape: (matched_points, 2)

    '''
    l = feat['l']
    r = feat['r']
    l_feature = l['feature']
    if objectness_mask is None:
        r_feature = r['feature']
        r_idx = torch.tensor(range(len(r['feature'])), device=r['feature'].device)
    else:
        obj_mask = objectness_mask.to(r['feature'].device).bool()
        r_feature = r['feature'][obj_mask]
        r_idx = torch.tensor(range(len(r['feature'])), device=r['feature'].device)[obj_mask]

    dists, indices = get_closest_point(l_feature, r_feature)  # (num_l_feature)
    if objectness_mask is not None:
        indices = r_idx[indices]

    if method == 'top':
        mask = dists < init_dist_thresh
        while torch.sum(mask) < 20:
            init_dist_thresh = iter_factor * init_dist_thresh
            mask = dists < init_dist_thresh
        print(f'final dist_thresh:{init_dist_thresh}')

    elif method == 'random':
        num_l = len(l_feature)
        num_match = min(num_l, match_number)
        ones_mask = np.ones(num_match, dtype=bool)
        zeros_mask = np.zeros(num_l - num_match, dtype=bool)
        match_mask = np.concatenate((ones_mask, zeros_mask))
        np.random.shuffle(match_mask)
        mask = torch.from_numpy(match_mask)
    else:
        raise ValueError(f'Unknown method for get inference matches:{method}')
    l_origin_index = torch.tensor(range(len(indices)), device=indices.device)[mask]
    r_origin_index = indices[mask]
    inference_matches = torch.stack([l_origin_index, r_origin_index]).T
    return inference_matches.cpu()


def get_inference_matches_from_correspondence(feat,
                                              correspondence_threshold=0.5,
                                              method='only_top',
                                              candidate_num=10):
    l = feat['l']
    r = feat['r']
    rgbxyz_l = l['rgbxyz']
    rgbxyz_r = r['rgbxyz']
    l_idx = l['idx']
    r_idx = r['idx']
    assert 'correspondence_map' in feat, 'correspondence map should be implemented to use this inference method'
    correspondence_map = feat['correspondence_map']

    inference_matches = []
    if method == 'only_top':
        max_correspondence = torch.max(correspondence_map, dim=1)
        max_correspondence_value = max_correspondence.values
        max_correspondence_idx = r_idx[max_correspondence.indices]
        for i in range(max_correspondence_idx.size(0)):
            if max_correspondence_value[i] < correspondence_threshold:
                continue
            inference_matches.append(torch.tensor([l_idx[i], max_correspondence_idx[i], max_correspondence_value[i]]))
    if method == 'top_and_rgb_match':
        correspondence_sorted_idx = torch.argsort(correspondence_map, dim=1)
        for i, l_i in enumerate(l_idx):
            candidates = correspondence_sorted_idx[i, -candidate_num:]
            dists, indices = get_closest_point(rgbxyz_l[i][3:].reshape(1, 3), rgbxyz_r[candidates][:, 3:])
            if correspondence_map[i, candidates[indices[0]]] > correspondence_threshold:
                inference_matches.append(torch.tensor([l_idx[i], r_idx[candidates[indices[0]]], correspondence_map[i, candidates[indices[0]]]]))

    if len(inference_matches) == 0:
        return torch.Tensor([])
    inference_matches = torch.stack(inference_matches)
    return inference_matches.cpu()


def get_correspondence_dist_heatmap(feat,
                                    correspondence_threshold=0.5,
                                    candidate_num=100):
    l = feat['l']
    r = feat['r']
    l_idx = l['idx']
    r_idx = r['idx']
    assert 'correspondence_map' in feat, 'correspondence map should be computed to use this inference method'
    correspondence_map = feat['correspondence_map']

    candidate_num = min(candidate_num, correspondence_map.size(0))

    res_dict = {}

    for i, r_i in enumerate(r_idx):
        correspondence_sorted_idx = torch.argsort(correspondence_map, dim=0)

        candidates = correspondence_sorted_idx[-candidate_num:, i]
        candidates = candidates[correspondence_map[candidates, i] > correspondence_threshold]

        res_dict[i] = {}
        res_dict[i]['source_point'] = i
        res_dict[i]['correspondence_points_candidates'] = candidates
        res_dict[i]['correspondence_points_confidence'] = correspondence_map[candidates, i]

    return res_dict
