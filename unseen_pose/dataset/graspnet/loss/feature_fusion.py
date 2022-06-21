import copy

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from ..graspnet_constant import RESIZE_HEIGHT, RESIZE_WIDTH, NUM_RGB_FEATURE, NUM_GEO_FEATURE, NUM_FUSION_FEATURE, NUM_RGBXYZ_FEATURE
# from ..graspnet_constant import RESIZE_HEIGHT, RESIZE_WIDTH, NUM_RGB_FEATURE, NUM_GEO_FEATURE, NUM_FUSION_FEATURE, NUM_RGBXYZ_FEATURE
from .semantic_segmentation import base3DPoseNet


class FeatureFusion(nn.Module):
    def __init__(self, num_rgb_feature = NUM_RGB_FEATURE, num_geo_feature = NUM_GEO_FEATURE, num_fusion_feature = NUM_FUSION_FEATURE):
        super().__init__()
        self.mlp = nn.Linear(num_rgb_feature + num_geo_feature, num_fusion_feature)

    def forward(self, feature_dict, batch_data_dict):
        '''
        feature_dict: dict of ['lr']['soutput', 'rgb_f']
        batch_data_dict: batch data returned by DataLoader.
        '''
        fused_feature_dict = dict()
        matches = batch_data_dict['matches']
        non_matches = batch_data_dict['non_matches']
        for lr_id, lr in enumerate(['l','r']):
            idxs = batch_data_dict[lr]['idxs'] # 
            start_idx = batch_data_dict[lr]['start_idx'] # 
            num_points = batch_data_dict[lr]['num_points'] #
            soutput = feature_dict[lr]['soutput'] # 
            soutput_c = soutput.C # (sum(num_points), 4)
            soutput_f = soutput.F # (sum(num_points), 512)
            all_rgb_feature = feature_dict[lr]['rgb_f'] # (b, 512, 288, 384)
            batch_num = len(num_points)
            matches_feature_list = []
            non_matches_feature_list = []
            for batch_id in range(batch_num):
                # if not len(start_idx) == 4:
                #     print(f'\033[31mWarning len(start_idx) = {len(start_idx)}, len(num_points) = {len(num_points)}\033[0m')
                # try:
                # print(f'batch_id:{batch_id}, start index:{start_idx[batch_id]}, end index:{start_idx[batch_id] + num_points[batch_id]}, idx length:{len(idxs)}, num_points_length:{len(num_points)}, start_idx_length:{len(start_idx)}')
                idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id])
                # except:
                
                # assert False
                # print(f"idx.shape:{idx.shape}")
                # batch_mask = soutput_c[:, 0] == batch_id
                geometry_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id], 512)
                # print(f"geometry_feature.shape:{geometry_feature.shape}")
                batch_matches = matches[batch_id, :, lr_id] # (num_matches)
                # print(f"batch_matches.shape:{batch_matches.shape}")
                batch_non_matches = non_matches[batch_id, :, lr_id] # (num_non_matches)
                # print(f"batch_non_matches.shape:{batch_non_matches.shape}")
                batch_rgb_feature = all_rgb_feature[batch_id].reshape(512, RESIZE_HEIGHT * RESIZE_WIDTH).transpose(0,1) # (288 * 384, 512)
                # print(f"batch_rgb_feature.shape:{batch_rgb_feature.shape}")
                ################ RGB Feature ################
                
                batch_matches_rgb_feature = batch_rgb_feature[batch_matches] # (num_matches, 512)
                # print(f"batch_matches_rgb_feature.shape:{batch_matches_rgb_feature.shape}")
                batch_non_matches_rgb_feature = batch_rgb_feature[batch_non_matches] # (num_non_matches, 512)
                # print(f"batch_non_matches_rgb_feature.shape:{batch_non_matches_rgb_feature.shape}")

                ################ Geo Feature ################
                batch_matches_index_in_idx = torch.searchsorted(idx, batch_matches) # (num_matches)
                # print(f"batch_matches_index_in_idx.shape:{batch_matches_index_in_idx.shape}")
                batch_non_matches_index_in_idx = torch.searchsorted(idx, batch_non_matches) # (num_non_matches)
                num_feature = len(geometry_feature)
                if torch.max(batch_matches_index_in_idx).item() >= num_feature:
                    print(f'\033[031m Error: matches index too big, max:{torch.max(batch_matches_index_in_idx).item()} feature:{num_feature}\033[0m')
                    print(batch_matches_index_in_idx)
                if torch.max(batch_non_matches_index_in_idx).item() >= num_feature:
                    print(f'\033[031m Error: non matches index too big, max:{torch.max(batch_non_matches_index_in_idx).item()} feature:{num_feature}\033[0m')
                    print(batch_non_matches_index_in_idx)
                    print(f'batch_non_matches:{batch_non_matches}')
                # print(f"batch_non_matches_index_in_idx.shape:{batch_non_matches_index_in_idx.shape}")
                batch_matches_geometry_feature = geometry_feature[batch_matches_index_in_idx] # (num_matches, 512)
                # print(f"batch_matches_geometry_feature.shape:{batch_matches_geometry_feature.shape}")
                batch_non_matches_geometry_feature = geometry_feature[batch_non_matches_index_in_idx] # (num_non_matches, 512)
                # print(f"batch_non_matches_geometry_feature.shape:{batch_non_matches_geometry_feature.shape}")

                ################ MLP Fusion ################
                batch_matches_cat_feature = torch.cat([batch_matches_rgb_feature, batch_matches_geometry_feature], dim = 1) # (num_matches, 1024)
                # print(f"batch_matches_cat_feature.shape:{batch_matches_cat_feature.shape}")
                batch_non_matches_cat_feature = torch.cat([batch_non_matches_rgb_feature, batch_non_matches_geometry_feature], dim = 1) # (num_non_matches, 1024)
                # print(f"batch_non_matches_cat_feature.shape:{batch_non_matches_cat_feature.shape}")
                batch_matches_feature = self.mlp(batch_matches_cat_feature) # (num_matches, 256)
                # print(f"batch_matches_feature.shape:{batch_matches_feature.shape}")
                batch_non_matches_feature = self.mlp(batch_non_matches_cat_feature) # (num_matches, 256)
                # print(f"batch_non_matches_feature.shape:{batch_non_matches_feature.shape}")

                matches_feature_list.append(batch_matches_feature)
                non_matches_feature_list.append(batch_non_matches_feature)
            fused_feature_dict[lr] = {
                'matches_feature': torch.stack(matches_feature_list), # (b, num_matches, 256)
                'non_matches_feature': torch.stack(non_matches_feature_list) # (b, num_non_matches, 256)
            }
        return fused_feature_dict

class FeatureFusionRGBXYZ(nn.Module):
    def __init__(self, num_rgbxyz_feature = NUM_RGBXYZ_FEATURE, num_fusion_feature = NUM_FUSION_FEATURE):
        super().__init__()
        self.num_rgbxyz_feature = num_rgbxyz_feature
        self.num_fusion_feature = num_fusion_feature
        # self.basePoseNet = base3DPoseNet(self.num_rgbxyz_feature, self.num_fusion_feature, use_batch_norm=False)
        # self.mlp_l = nn.Linear(self.num_rgbxyz_feature, self.num_fusion_feature)
        self.basePoseNet = base3DPoseNet(self.num_rgbxyz_feature * 2, 2, use_batch_norm=False)

    def forward(self, feature_dict, batch_data_dict):
        '''
        feature_dict: dict of ['lr']['soutput', 'rgb_f']
        batch_data_dict: batch data returned by DataLoader.
        '''
        fused_feature_dict = dict()

        # print("data batch in feature fusion, ", 64 * "-")
        # print(batch_data_dict['l'])
        # print(batch_data_dict['r'])

        matches = batch_data_dict['matches']
        non_matches = batch_data_dict['non_matches']
        for lr_id, lr in enumerate(['l','r']):
            idxs = batch_data_dict[lr]['idxs'] # 
            start_idx = batch_data_dict[lr]['start_idx'] # 
            num_points = batch_data_dict[lr]['num_points'] #
            soutput = feature_dict[lr]['soutput'] # 
            soutput_c = soutput.C # (sum(num_points), 4)
            soutput_f = soutput.F # (sum(num_points), 512)
            batch_num = len(num_points)
            matches_feature_list = []
            non_matches_feature_list = []

            for batch_id in range(batch_num):
                # print(idxs.shape)
                # print(start_idx[batch_id])
                # print(num_points[batch_id])

                # idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id])
                rgbxyz_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id], 512)

                # print(matches.shape)
                batch_matches = matches[batch_id, :, lr_id] # (num_matches)
                batch_matches = batch_matches[batch_matches > -1]
                # print("real batch matches:, " , batch_matches.shape)
                batch_non_matches = non_matches[batch_id, :, lr_id] # (num_non_matches)
                batch_non_matches = batch_non_matches[batch_non_matches > -1]
                # print("real batch non matches:, " , batch_non_matches.shape)
                batch_matches_rgbxyz_feature = rgbxyz_feature[batch_matches] # (num_matches, 512)
                batch_non_matches_rgbxyz_feature = rgbxyz_feature[batch_non_matches] # (num_non_matches, 512)

                ################ MLP Fusion ################
                # batch_matches_feature = self.basePoseNet(batch_matches_rgbxyz_feature) # (num_matches, 256)
                # batch_non_matches_feature = self.basePoseNet(batch_non_matches_rgbxyz_feature) # (num_matches, 256)

                ############### DIRECT MLP PREDICTION ###############
                batch_matches_feature = batch_matches_rgbxyz_feature # (num_matches, 512)
                batch_non_matches_feature = batch_non_matches_rgbxyz_feature # (num_matches, 512)

                matches_feature_list.append(batch_matches_feature)
                non_matches_feature_list.append(batch_non_matches_feature)

            fused_feature_dict[lr] = {
                # 'matches_feature': torch.stack(matches_feature_list), # (b, num_matches, 512)
                # 'non_matches_feature': torch.stack(non_matches_feature_list) # (b, num_non_matches, 512)
                'matches_feature': matches_feature_list, # b * ( num_matches, 512)
                'non_matches_feature': non_matches_feature_list # b *( num_non_matches, 512)
            }

        # matches_feature = torch.cat((fused_feature_dict['l']['matches_feature'], fused_feature_dict['r']['matches_feature']), dim=2)  # (b, num_matches, 1024)
        # non_matches_feature = torch.cat((fused_feature_dict['l']['non_matches_feature'], fused_feature_dict['r']['non_matches_feature']), dim=2) # (b, num_non_matches, 1024)

        matches_feature_l = torch.cat(fused_feature_dict['l']['matches_feature'], dim=0)  # (num_matches, 512))
        matches_feature_r = torch.cat(fused_feature_dict['r']['matches_feature'], dim=0)
        non_matches_feature_l = torch.cat(fused_feature_dict['l']['non_matches_feature'], dim=0)
        non_matches_feature_r = torch.cat(fused_feature_dict['r']['non_matches_feature'], dim=0)

        matches_feature = torch.cat((matches_feature_l, matches_feature_r), dim=1)  # (num_matches, 1024)
        non_matches_feature = torch.cat((non_matches_feature_l, non_matches_feature_r), dim=1)

        matches_label = torch.ones(matches_feature.size(0)) # (num_matches, )
        non_matches_label = torch.zeros(non_matches_feature.size(0))

        feature = torch.cat((matches_feature, non_matches_feature), dim=0)  # concatenate the features
        feature = feature.reshape((-1, feature.size(1)))  # concatenate on the batch dimension

        label = torch.cat((matches_label, non_matches_label)).to(feature.device)
        prediction = self.basePoseNet(feature)

        fused_feature_dict['match_label'] = label
        fused_feature_dict['match_pred'] = prediction

        return fused_feature_dict


class InferenceFeatureFusion(nn.Module):
    def __init__(self, num_rgb_feature = NUM_RGB_FEATURE, num_geo_feature = NUM_GEO_FEATURE, num_fusion_feature = NUM_FUSION_FEATURE):
        super().__init__()
        self.mlp_l = nn.Linear(num_rgb_feature + num_geo_feature, num_fusion_feature)

    def forward(self, feature_dict, batch_data_dict):
        '''
        feature_dict: dict of ['lr']['soutput', 'rgb_f']
        batch_data_dict: batch data returned by DataLoader.
        '''
        fused_feature_dict = dict()
        for lr_id, lr in enumerate(['l','r']):
            idxs = batch_data_dict[lr]['idxs'] # 
            start_idx = batch_data_dict[lr]['start_idx'] # 
            num_points = batch_data_dict[lr]['num_points'] #
            soutput = feature_dict[lr]['soutput'] # 
            soutput_c = soutput.C # (sum(num_points), 4)
            soutput_f = soutput.F # (sum(num_points), 512)
            all_rgb_feature = feature_dict[lr]['rgb_f'] # (b, 512, 288, 384)
            batch_num = len(num_points)
            assert batch_num == 1, 'batch_num should be one'
            batch_id = 0
            idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id])
            batch_geometry_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id], 512)
            output_c = soutput_c[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id], 4)
            batch_rgb_feature = all_rgb_feature[batch_id].reshape(512, RESIZE_HEIGHT * RESIZE_WIDTH).transpose(0,1) 
            batch_rgb_feature_in_index = batch_rgb_feature[idx] # (num_points[batch_id])
            batch_cat_feature = torch.cat([batch_rgb_feature_in_index, batch_geometry_feature], dim = 1) # (num_matches, 1024)
            batch_feature = self.mlp_l(batch_cat_feature) # (num_matches, 256)
            ################ RGB Feature ################

            fused_feature_dict[lr] = {
                'feature': batch_feature, # (num_points, 256)
                'idx': idx, # (num_points)
            }
        return fused_feature_dict


class InferenceFeatureFusionRGBXYZ(nn.Module):
    def __init__(self, num_rgbxyz_feature = NUM_RGBXYZ_FEATURE, num_fusion_feature = NUM_FUSION_FEATURE):
        super().__init__()
        self.num_rgbxyz_feature = num_rgbxyz_feature
        self.num_fusion_feature = num_fusion_feature
        # self.basePoseNet = base3DPoseNet(self.num_rgbxyz_feature, self.num_fusion_feature, use_batch_norm=False)
        self.basePoseNet = base3DPoseNet(self.num_rgbxyz_feature * 2, 2, use_batch_norm=False)
        # self.mlp_l = nn.Linear(self.num_rgbxyz_feature, self.num_fusion_feature)

    def forward(self, feature_dict, batch_data_dict, sample_number_r=64, sample_number_l=1600, mask=None):
        '''
        feature_dict: dict of ['lr']['soutput', 'rgb_f']
        batch_data_dict: batch data returned by DataLoader.
        '''
        fused_feature_dict = dict()

        sample_number = {
            'l': sample_number_l,
            'r': sample_number_r
        }

        for lr_id, lr in enumerate(['l','r']):
            idxs = batch_data_dict[lr]['idxs'] # 
            start_idx = batch_data_dict[lr]['start_idx'] # 
            num_points = batch_data_dict[lr]['num_points'] #
            rgbxyz = batch_data_dict[lr]['points_batch']
            soutput = feature_dict[lr]['soutput'] # 
            # soutput_c = soutput.C # (sum(num_points), 4)
            soutput_f = soutput.F # (sum(num_points), 512)
            batch_num = len(num_points)
            assert batch_num == 1, 'batch num should be one'
            batch_id = 0
            idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id])
            batch_rgbxyz_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]] # (num_points[batch_id], 512)
            batch_rgbxyz = rgbxyz[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
            if lr == 'r':
                # using a mask to select the target object
                if mask is None:
                    objectness_labels = torch.ones(batch_data_dict[lr]['sem_seg_label'].shape)
                    objectness_label = objectness_labels[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
                    mask = objectness_label
                idx = torch.arange(0, batch_rgbxyz.size(0), 1)[mask > 0.5]
                idx_mask = copy.deepcopy(idx)
                sample_number['r'] = min(idx.size(0), sample_number['r'])
                sample_idx = idx[torch.randint(0, idx.size(0), (sample_number['r'],))]
                # batch_rgbxyz_feature_tsne = batch_rgbxyz_feature[idx_mask]
                batch_rgbxyz_feature_tsne = batch_rgbxyz_feature[::5]
                # batch_rgbxyz_tsne = batch_rgbxyz[idx_mask]
                batch_rgbxyz_tsne = batch_rgbxyz[::5]
                batch_rgbxyz_feature = batch_rgbxyz_feature[sample_idx]
                batch_rgbxyz = batch_rgbxyz[sample_idx]
                idx = sample_idx
            else:
                # random sample a number of points
                idx = torch.arange(0, batch_rgbxyz.size(0), 1)
                sample_number['l'] = min(idx.size(0), sample_number['l'])
                sample_idx = idx[torch.randint(0, idx.size(0), (sample_number['l'],))]
                batch_rgbxyz_feature_tsne = batch_rgbxyz_feature[::5]
                batch_rgbxyz_tsne = batch_rgbxyz[::5]
                batch_rgbxyz_feature = batch_rgbxyz_feature[sample_idx]
                batch_rgbxyz = batch_rgbxyz[sample_idx]
                idx = sample_idx

            batch_feature = batch_rgbxyz_feature

            fused_feature_dict[lr] = {
                'feature_tsne': batch_rgbxyz_feature_tsne,
                'rgbxyz_tsne': batch_rgbxyz_tsne,
                'feature': batch_feature, # (num_sample(or masked), 512)
                'rgbxyz': batch_rgbxyz, # (num_sample(or masked), 6)
                'idx': idx # (num_sample(or masked))
            }

        correspondence_map = torch.empty((fused_feature_dict['l']['idx'].size(0), fused_feature_dict['r']['idx'].size(0)))
        for i in range(sample_number['r']):
            feature_cat = torch.cat(
                (
                    fused_feature_dict['l']['feature'],
                    fused_feature_dict['r']['feature'][i].repeat((fused_feature_dict['l']['feature'].size(0), 1)),
                ), dim=1
            )  # (num_masked, 1024)
            correspondence_predict = self.basePoseNet(feature_cat)
            correspondence_predict = nn.functional.softmax(correspondence_predict, dim=1)
            correspondence_map[:, i] = correspondence_predict[:, 1]
        fused_feature_dict['correspondence_map'] = correspondence_map
        # print(correspondence_map)
        return fused_feature_dict
