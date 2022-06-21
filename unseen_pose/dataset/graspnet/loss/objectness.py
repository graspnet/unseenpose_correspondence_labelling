import torch
import torch.nn as nn
import MinkowskiEngine as ME

from ..graspnet_constant import RESIZE_HEIGHT, RESIZE_WIDTH, NUM_RGB_FEATURE, NUM_GEO_FEATURE, NUM_FUSION_FEATURE, NUM_RGBXYZ_FEATURE


class ObjectnessNet(nn.Module):
    """
            detect if the point is object or the back ground from its RGBXYZ feature
    """
    def __init__(self, num_rgbxyz_feature = NUM_RGBXYZ_FEATURE, num_fusion_feature = NUM_FUSION_FEATURE):
        super().__init__()
        self.num_rgbxyz_feature = num_rgbxyz_feature
        self.num_fusion_feature = num_fusion_feature
        self.mlp_1 = nn.Linear(self.num_rgbxyz_feature, self.num_fusion_feature)
        self.mlp_2 = nn.Linear(self.num_fusion_feature, self.num_fusion_feature)
        self.mlp_3 = nn.Linear(self.num_fusion_feature, self.num_fusion_feature)
        self.mlp_4 = nn.Linear(self.num_fusion_feature, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_dict, batch_data_dict, num_sample=200):
        lr = 'r'

        ret_dict = dict()
        idxs = batch_data_dict[lr]['idxs'] #
        start_idx = batch_data_dict[lr]['start_idx'] #
        num_points = batch_data_dict[lr]['num_points'] #
        objectness_labels = batch_data_dict[lr]['objectness_label'] #

        soutput = feature_dict[lr]['soutput'] #
        soutput_c = soutput.C # (sum(num_points), 4)
        soutput_f = soutput.F # (sum(num_points), 512)
        batch_num = len(num_points)

        obj_pred_list = []
        obj_label_list = []
        for batch_id in range(batch_num):
            idx = idxs[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
            rgbxyz_feature = soutput_f[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]
            objectness_label = objectness_labels[start_idx[batch_id]:start_idx[batch_id] + num_points[batch_id]]

            sample_idx = torch.randint(rgbxyz_feature.size()[0], (num_sample,), dtype=torch.int64)

            batch_rgbxyz_feature = rgbxyz_feature[sample_idx]
            batch_objectness_label = objectness_label[sample_idx]

            batch_objectness_pred = self.softmax(self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(batch_rgbxyz_feature)))))

            obj_pred_list.append(batch_objectness_pred)
            obj_label_list.append(batch_objectness_label)

        ret_dict['objectness_pred'] = torch.stack(obj_pred_list)
        ret_dict['objectness_label'] = torch.stack(obj_label_list)

        return ret_dict
