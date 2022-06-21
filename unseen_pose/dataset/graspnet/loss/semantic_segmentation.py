import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from ..graspnet_constant import RESIZE_HEIGHT, RESIZE_WIDTH, NUM_RGB_FEATURE, NUM_GEO_FEATURE, NUM_FUSION_FEATURE, \
    NUM_RGBXYZ_FEATURE


class base3DPoseNet(nn.Module):
    """
    block of mlps for feature transformation for further prediction (?)
    """

    def __init__(self,
                 num_feature,
                 num_out,
                 num_fusion_feature=1024, dropout_keep_prob=0.5, block_num=2, use_norm=False):
        super().__init__()
        self.num_feature = num_feature
        self.num_out = num_out
        self.num_fusion_feature = num_fusion_feature
        self.dropout_keep_prob = dropout_keep_prob
        self.block_num = block_num
        self.use_norm = use_norm
        self.network_initialization()

    def linear_block(self):
        if self.use_norm:
            return nn.Sequential(
                nn.Linear(self.num_fusion_feature, self.num_fusion_feature),
                nn.LayerNorm(self.num_fusion_feature),
                nn.ReLU(),
                nn.Dropout(1 - self.dropout_keep_prob),
            )
        return nn.Sequential(
            nn.Linear(self.num_fusion_feature, self.num_fusion_feature),
            nn.ReLU(),
            nn.Dropout(1 - self.dropout_keep_prob),
        )

    def network_initialization(self):
        self.input = nn.Linear(self.num_feature, self.num_fusion_feature)

        self.block11 = self.linear_block()
        self.block12 = self.linear_block()
        self.block21 = self.linear_block()
        self.block22 = self.linear_block()

        self.output = nn.Linear(self.num_fusion_feature, self.num_out)

    def forward(self, xin, residual=True):
        xin = self.input(xin)

        x1 = self.block11(xin)
        if residual:
            x1 = x1 + xin
        x2 = self.block12(x1)
        if residual:
            x2 = x2 + x1
            x2 = x2 + xin

        x3 = self.block21(x2)
        if residual:
            x3 = x3 + x2
        x4 = self.block22(x3)
        if residual:
            x4 = x4 + x3
        return self.output(x4)


class UnseenSemSegNet(nn.Module):
    """
            detect if the points in the right point cloud which belongs to the left object
    """

    def __init__(self, num_rgbxyz_feature=NUM_RGBXYZ_FEATURE, num_fusion_feature=NUM_FUSION_FEATURE,
                 method='mlp'):
        super().__init__()
        self.num_rgbxyz_feature = num_rgbxyz_feature
        self.num_fusion_feature = num_fusion_feature
        self.mlp_l = nn.Linear(num_rgbxyz_feature, num_fusion_feature)
        self.mlp_r = nn.Linear(num_rgbxyz_feature, num_fusion_feature)
        self.relu = nn.ReLU()

        assert method in ['mlp', '3dpose'], f'{method} not supported'
        if method == 'mlp':
            self.sem_seg_net = nn.Sequential(
                nn.Linear(2 * num_fusion_feature, num_fusion_feature),
                nn.ReLU(),
                nn.Linear(num_fusion_feature, num_fusion_feature),
                nn.ReLU(),
                # nn.Linear(num_fusion_feature, num_fusion_feature),
                # nn.ReLU(),
                nn.Linear(num_fusion_feature, 2)
            )
        else:
            self.sem_seg_net = base3DPoseNet(2 * num_fusion_feature, num_out=2)

    def balance_sample(self, label, obj_label, num_sample, total_batch):
        """
        Balance sample the positive and negative samples from the scene

        """
        pos_ratio = max(0.35 - 5 * (total_batch) / (6 * 10 ** 6), 0.1)
        # pos_ratio = 0.1 + 0.25 * np.random.rand()
        neg_obj_ratio = min(0.45 + 5 * (total_batch) / (6 * 10 ** 6), 0.7)
        # neg_obj_ratio = 0.8 - pos_ratio

        pos_mask = label > 0.5
        pos_idx = torch.arange(label.size(0))[pos_mask]
        pos_sample_num = min(pos_idx.size(0), int(pos_ratio * num_sample))

        neg_mask = label <= 0.5
        obj_mask = obj_label > 0.5

        neg_obj_idx = torch.arange(label.size(0))[torch.logical_and(neg_mask, obj_mask)]
        neg_obj_sample_num = min(neg_obj_idx.size(0), int(neg_obj_ratio * num_sample))
        neg_obj_sample_idx = neg_obj_idx[torch.randint(neg_obj_idx.size(0), (neg_obj_sample_num,), dtype=torch.int64)]

        neg_back_idx = torch.arange(label.size(0))[torch.logical_and(neg_mask, torch.logical_not(obj_mask))]
        neg_back_sample_idx = neg_back_idx[
            torch.randint(neg_back_idx.size(0), (num_sample - pos_sample_num - neg_obj_sample_num,), dtype=torch.int64)]

        if pos_sample_num <= 0:
            print('all negative samples')  # DEBUG MESSAGE
            return torch.cat((neg_obj_sample_idx, neg_back_sample_idx))

        pos_sample_idx = pos_idx[torch.randint(pos_idx.size(0), (pos_sample_num,), dtype=torch.int64)]
        sample_idx = torch.cat((pos_sample_idx, neg_obj_sample_idx, neg_back_sample_idx))
        return sample_idx[torch.randperm(sample_idx.size(0))]  # shuffle the sample indexes

    def forward(self, feature_dict, batch_data_dict, num_sample=700, total_batch=0):
        ret_dict = dict()

        sem_seg_label_list = []
        sem_seg_feature_r_list = []
        sem_seg_feature_l_list = []
        sem_seg_idx_list = []

        for lr_id, lr in enumerate(['l', 'r']):
            idxs = batch_data_dict[lr]['idxs']  #
            start_idx = batch_data_dict[lr]['start_idx']  #
            num_points = batch_data_dict[lr]['num_points']  #
            sem_seg_labels = batch_data_dict[lr]['sem_seg_label']  #
            objectness_labels = batch_data_dict[lr]['objectness_label']  #
            color = batch_data_dict[lr]['points_batch'][:, 3:]
            # print("points batch: ", color.size())
            soutput = feature_dict[lr]['soutput']  #
            soutput_c = soutput.C  # (sum(num_points), 4)
            soutput_f = soutput.F  # (sum(num_points), 512)
            batch_num = len(num_points)

            for batch_id in range(batch_num):
                idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
                rgbxyz_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]

                if lr == 'r':
                    sem_seg_label = sem_seg_labels[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
                    objectness_label = objectness_labels[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
                    # print(sem_seg_label.sum() , sem_seg_label.size())
                    # sample_idx = torch.randint(rgbxyz_feature.size()[0], (num_sample,), dtype=torch.int64)
                    sample_idx = self.balance_sample(sem_seg_label, objectness_label, num_sample,
                                                     total_batch=total_batch)
                    batch_idx = sample_idx
                    batch_rgbxyz_feature = rgbxyz_feature[sample_idx]
                    batch_sem_seg_label = sem_seg_label[sample_idx]
                    sem_seg_label_list.append(batch_sem_seg_label)
                    sem_seg_feature_r_list.append(self.mlp_r(batch_rgbxyz_feature))
                    sem_seg_idx_list.append(batch_idx)
                else:
                    obj_avg_feature = torch.mean(rgbxyz_feature, dim=0).repeat((num_sample, 1))
                    # print(obj_avg_feature[0])
                    obj_avg_feature_fusion = self.mlp_l(obj_avg_feature)
                    sem_seg_feature_l_list.append(obj_avg_feature_fusion)

        sem_seg_pred_list = []
        for sem_seg_feature_l, sem_seg_feature_r in zip(sem_seg_feature_l_list, sem_seg_feature_r_list):
            sem_seg_feature = torch.cat((sem_seg_feature_l, sem_seg_feature_r), dim=1)
            sem_seg_pred_list.append(self.sem_seg_net(sem_seg_feature))

        ret_dict['sem_seg_pred'] = torch.stack(sem_seg_pred_list)
        ret_dict['sem_seg_label'] = torch.stack(sem_seg_label_list)
        ret_dict['idx'] = torch.stack(sem_seg_idx_list)
        return ret_dict


class InferenceUnseenSemSegNet(nn.Module):
    """
            detect if the points in the right point cloud which belongs to the left object (Inference version)
    """
    def __init__(self, num_rgbxyz_feature=NUM_RGBXYZ_FEATURE,
                 num_fusion_feature=NUM_FUSION_FEATURE,
                 method=None):
        super().__init__()
        self.num_rgbxyz_feature = num_rgbxyz_feature
        self.num_fusion_feature = num_fusion_feature
        self.relu = nn.ReLU()
        self.mlp_l = nn.Linear(num_rgbxyz_feature, num_fusion_feature)
        self.mlp_r = nn.Linear(num_rgbxyz_feature, num_fusion_feature)
        assert method in ['mlp', '3dpose'], f'{method} not supported'
        if method == 'mlp':
            self.sem_seg_net = nn.Sequential(
                nn.Linear(2 * num_fusion_feature, num_fusion_feature),
                nn.ReLU(),
                nn.Linear(num_fusion_feature, num_fusion_feature),
                nn.ReLU(),
                # nn.Linear(num_fusion_feature, num_fusion_feature),
                # nn.ReLU(),
                nn.Linear(num_fusion_feature, 2)
            )
        else:
            self.sem_seg_net = base3DPoseNet(2 * num_fusion_feature, 2)

    def forward(self, feature_dict, batch_data_dict, num_sample=100000, total_batch=0):
        '''
        feature_dict: dict of ['lr']['soutput', 'rgb_f']
        batch_data_dict: batch data returned by DataLoader.
        '''
        sem_seg_feature_r_list = []
        sem_seg_feature_l_list = []

        for lr_id, lr in enumerate(['l', 'r']):
            idxs = batch_data_dict[lr]['idxs']  #
            start_idx = batch_data_dict[lr]['start_idx']  #
            num_points = batch_data_dict[lr]['num_points']  #

            # print(idxs.shape)
            soutput = feature_dict[lr]['soutput']  #
            soutput_c = soutput.C  # (sum(num_points), 4)
            soutput_f = soutput.F  # (sum(num_points), 512)
            batch_num = len(num_points)

            assert batch_num == 1, 'batch num should be one for inference'

            batch_id = 0
            idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
            rgbxyz_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]

            if lr == 'r':
                sample_i = torch.randint(idx.size(0), (num_sample,), dtype=torch.int64)

                # batch_idx = idx[sample_i]
                batch_idx = sample_i
                batch_rgbxyz_feature = rgbxyz_feature[sample_i]
                batch_rgbxyz_feature_fusion = self.mlp_r(batch_rgbxyz_feature)
            else:
                obj_avg_feature = torch.mean(rgbxyz_feature, dim=0).repeat((num_sample, 1))
                # print(obj_avg_feature[0])
                obj_avg_feature_fusion = self.mlp_l(obj_avg_feature)

        sem_seg_feature = torch.cat((obj_avg_feature_fusion, batch_rgbxyz_feature_fusion), dim=1)
        # print(sem_seg_feature[:10])
        sem_seg_pred = self.sem_seg_net(sem_seg_feature)
        # print(sem_seg_pred.size())
        return {
            'idx': batch_idx,
            'semantic segmentation result': torch.nn.functional.softmax(sem_seg_pred, dim=1),
        }
