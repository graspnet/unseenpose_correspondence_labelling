import torch
import torch.nn as nn
import MinkowskiEngine as ME
from ..dataset.graspnet.graspnet_constant import NUM_FUSION_FEATURE, NUM_GEO_FEATURE
from .fastpose import FastPose
from .resunet import ResUNet14
from ..dataset.graspnet.loss.feature_fusion import FeatureFusion, InferenceFeatureFusion, FeatureFusionRGBXYZ, InferenceFeatureFusionRGBXYZ
from ..dataset.graspnet.loss.objectness import ObjectnessNet
from ..dataset.graspnet.loss.semantic_segmentation import UnseenSemSegNet, InferenceUnseenSemSegNet
from ..dataset.graspnet.graspnet_constant import NUM_RGBXYZ_FEATURE
from ..get_boundingbox_from_segmentation import get_points_in_box_from_point_cloud, \
    get_contoured_points_from_bounding_box, get_heatmap_from_point_cloud, get_contour_from_heat_map


class UnifiedFeatureNet(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.geometry_net = ResUNet14(in_channels=3, out_channels=NUM_GEO_FEATURE, conv1_kernel_size=3, bn_momentum=0.02, D=3)
        self.color_net = FastPose()
        self.color_upsample = nn.Upsample(scale_factor = 4, mode = 'bilinear')
        self.inference = inference
        if self.inference:
            self.ff_layer = InferenceFeatureFusion()
        else:
            self.ff_layer = FeatureFusion()

    def forward(self, x):
        '''
        **Input:**

        - x is a dict returned by the data Loader

        - 'sinput': sparse tensor of point cloud.

        - 'index': index of selected point.

        - 'rgb': input rgb image. (288, 384)
        
        **Output:**

        - concatenated features of each point of shape (batchsize, num_points, num_shape_feature + num_color_feature).
        '''
        feature_dict = dict() 
        for lr in ['l', 'r']:
            x[lr]['sinput'] = ME.SparseTensor(x[lr]['points_batch'], x[lr]['coords_batch'], device = x[lr]['points_batch'].device)
            soutput = self.geometry_net(x[lr]['sinput'])
            rgb_f = self.color_upsample(self.color_net(x[lr]['rgb']))
            feature_dict[lr] = {'soutput':soutput, 'rgb_f':rgb_f}
        
        return self.ff_layer(feature_dict, x)


class RGBXYZNet(nn.Module):
    def __init__(self, inference=False, input_method='rgbxyz', bn_momentum=0.02,
                 freeze_backbone=False,
                 freeze_segmentation_block=False,
                 freeze_matching_block=False):
        super().__init__()
        self.inference = inference
        self.input_method = input_method
        if input_method == 'rgbxyz':
            self.feature_net = ResUNet14(in_channels=6, out_channels=NUM_RGBXYZ_FEATURE, conv1_kernel_size=3, bn_momentum=bn_momentum, D=3)
        elif input_method == 'rgb' or input_method == 'xyz':
            self.feature_net = ResUNet14(in_channels=3, out_channels=NUM_RGBXYZ_FEATURE, conv1_kernel_size=3, bn_momentum=bn_momentum, D=3)
        else:
            raise ValueError('RGBXYZ net input unknown method!')

        if freeze_backbone:
            for param in self.feature_net.parameters():
                param.requires_grad = False

        if self.inference:
            self.ff_layer = InferenceFeatureFusionRGBXYZ()
            self.sem_seg_net = InferenceUnseenSemSegNet(method='3dpose')
        else:
            self.ff_layer = FeatureFusionRGBXYZ()
            self.sem_seg_net = UnseenSemSegNet(method='3dpose')

        if freeze_segmentation_block:
            for param in self.sem_seg_net.parameters():
                param.requires_grad = False
        if freeze_matching_block:
            for param in self.ff_layer.parameters():
                param.requires_grad = False


        # self.color_net = FastPose()
        # self.color_upsample = nn.Upsample(scale_factor = 4, mode = 'bilinear')

    def forward(self, x, total_batch=0):
        '''
        **Input:**

        - x is a dict returned by the data Loader

        - 'sinput': sparse tensor of point cloud.

        - 'index': index of selected point.

        - 'rgb': input rgb image. (288, 384)
        
        **Output:**

        - concatenated features of each point of shape (batchsize, num_points, num_shape_feature + num_color_feature).
        '''

        feature_dict = dict() 
        for lr in ['l', 'r']:
            # print('test shape:')
            # print(x[lr]['coords_batch'].shape)
            if self.input_method == 'rgbxyz':
                x[lr]['sinput'] = ME.SparseTensor(x[lr]['points_batch'], x[lr]['coords_batch'], device = x[lr]['points_batch'].device)
            elif self.input_method == 'rgb':
                x[lr]['sinput'] = ME.SparseTensor(x[lr]['points_batch'][:,3:], x[lr]['coords_batch'], device = x[lr]['points_batch'].device)
            elif self.input_method == 'xyz':
                x[lr]['sinput'] = ME.SparseTensor(x[lr]['points_batch'][:,:3], x[lr]['coords_batch'], device = x[lr]['points_batch'].device)
            # if lr == 'r':
            #     print(64 * '-=')
            #     print('sinput:', lr)
            #     print(x[lr]['sinput'])
            soutput = self.feature_net(x[lr]['sinput'])
            feature_dict[lr] = {'soutput':soutput}

        if not self.inference:
            return self.ff_layer(feature_dict, x), self.sem_seg_net(feature_dict, x, total_batch=total_batch)

        # IMPLEMENT THE BOUNDING BOX SELECTION AND CREATE A LIST OF REGION POSSIBLE
        else:
            obj_res = self.sem_seg_net(feature_dict, x, total_batch=total_batch)

            sem_seg_res = obj_res['semantic segmentation result']
            sem_seg_idx = obj_res['idx']
            pos_idxs = sem_seg_idx[sem_seg_res[:, 1] > sem_seg_res[:, 0]]
            rgbxyz_r = x['r']['points_batch']
            rgbxyz_segmented = rgbxyz_r[pos_idxs]
            # print(x['repr'])
            # print(rgbxyz_r.mean(dim=0), rgbxyz_r.std(dim=0))
            # print(rgbxyz_segmented.mean(dim=0), rgbxyz_segmented.std(dim=0))

            bboxes, heatmap = get_points_in_box_from_point_cloud(rgbxyz_r.detach().cpu().numpy().copy(),
                                                        rgbxyz_segmented.detach().cpu().numpy().copy())
            matching_res = []

            for bbox_points, bbox_mask in bboxes:
                # print(bbox_points.shape)
                # print(bbox_mask.shape)
                bbox_mask = torch.from_numpy(bbox_mask).to(x['r']['points_batch'].device)
                bbox_points = torch.from_numpy(bbox_points).to(x['r']['points_batch'].device)

                #TO DO: JUST FOR ABLATION STUDY
                matching_ = self.ff_layer(feature_dict, x, mask=bbox_mask)
                # matching_ = self.ff_layer(feature_dict, x, mask=None)
                matching_['bbox_points'] = bbox_points
                matching_res.append(matching_)

            # sem_seg_mask = torch.zeros(rgbxyz_r.size(0))
            # sem_seg_mask[pos_idxs] = 1.0
            return matching_res, obj_res, heatmap

