import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME

from .resunet import ResUNet14
from loss import process_grasp_labels, match_grasp_view_and_label, get_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
import pointnet2_utils
from pytorch_utils import SharedMLP
from pt_utils import generate_grasp_views, batch_viewpoint_params_to_matrix

class PointCylinderGroup(nn.Module):
    """Point Cylinder Group modules"""
    def __init__(self, nsample, mlps, cylinder_radius=0.05, hmin=-0.02, hmax=0.04, bn=True):
        super().__init__()
        self.cylinder_radius = cylinder_radius
        self.nsample = nsample

        mlps[0] += 3
        self.grouper = pointnet2_utils.CylinderQueryAndGroup(
            cylinder_radius, hmin, hmax, nsample, use_xyz=True, normalize_xyz=True, rotate_xyz=True
        )
        self.mlps = SharedMLP(mlps, bn=bn)

    def forward(self, xyz, new_xyz, view_rot, features):
        """
        xyz: (batch_size, num_point, 3)
        new_xyz: (batch_size, num_seed, 3)
        seed_features: (batch_size, feature_dim, num_point)
        vp_rot: (batch_size, num_seed, 3, 3)
        """
        B, num_seed, _, _ = view_rot.size()
        grouped_features = self.grouper(
            xyz, new_xyz, view_rot, features
        ) # (batch_size, feature_dim, num_seed, num_sample)
        new_features = self.mlps(
            grouped_features
        ) # (batch_size, mlps[-1], num_seed*view_factor*num_depth, nsample)
        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        ) # (batch_size, mlps[-1], num_seed, 1)
        new_features = new_features.squeeze(3) # (batch_size, mlps[-1], num_seed)

        return new_features

class ViewEstimator(nn.Module):
    def __init__(self, in_channels=512, num_samples=1024, num_view=300, sampling='heatmap_fps', heatmap_th=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.num_view = num_view
        self.sampling = sampling
        self.heatmap_th = heatmap_th

        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.conv3 = nn.Conv1d(in_channels, num_view, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, sfeat, cloud_list, obj_mask=None, heatmap=None):
        # sample points and features
        if 'heatmap' in self.sampling:
            assert(obj_mask is not None)
            assert(heatmap is not None)
        coords, feats = sfeat.C, sfeat.F
        seed_xyz = []
        seed_inds = []
        seed_features = []
        for i in range(len(cloud_list)):
            cloud_mask = (coords[:,0] == i)
            seed_inds_i = self.sample_grasp_points(cloud_list[i], obj_mask[cloud_mask], heatmap[cloud_mask])
            seed_xyz.append(cloud_list[i][seed_inds_i])
            seed_inds.append(seed_inds_i)
            seed_features.append(feats[cloud_mask][seed_inds_i])
        seed_xyz = torch.stack(seed_xyz, dim=0) #(B, Ns, 3)
        seed_inds = torch.stack(seed_inds, dim=0) #(B, Ns)
        seed_features = torch.stack(seed_features, dim=0)
        seed_features = seed_features.transpose(1, 2).contiguous() #(B, C, Ns)

        # forward pass
        out = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        seed_features = out + seed_features
        out = self.conv3(out)
        out = out.transpose(1, 2) #(B, Ns, V)

        return out, seed_xyz, seed_inds, seed_features.detach()

    def sample_grasp_points(self, points, obj_mask=None, heatmap=None):
        if 'heatmap' in self.sampling:
            assert(obj_mask is not None)
            assert(heatmap is not None)
            obj_inds = torch.where(obj_mask)[0]
            seed_inds = torch.where(obj_mask & (heatmap>self.heatmap_th))[0]
            if seed_inds.size(0) <= self.num_samples:
                if obj_inds.size(0) <= self.num_samples:
                    seed_inds = torch.argsort(heatmap, descending=True)[:self.num_samples]
                else:
                    seed_inds = torch.argsort(heatmap[obj_inds], descending=True)[:self.num_samples]
                    seed_inds = obj_inds[seed_inds]
                return seed_inds
        if self.sampling == 'heatmap_fps':
            fps_inds = pointnet2_utils.furthest_point_sample(points[seed_inds].unsqueeze(0), self.num_samples)
            fps_inds = fps_inds.squeeze(0).long()
            seed_inds = seed_inds[fps_inds]
        elif self.sampling == 'heatmap_random':
            rand_inds = torch.randperm(seed_inds.size(0))[:self.num_samples]
            seed_inds = seed_inds[rand_inds]
        elif self.sampling == 'fps':
            fps_inds = pointnet2_utils.furthest_point_sample(points.unsqueeze(0), self.num_samples)
            seed_inds = fps_inds.squeeze(0).long()
        elif self.sampling == 'random':
            seed_inds = torch.randperm(points.size(0))[:self.num_samples]
        else:
            print('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        return seed_inds

class GraspGenerator(nn.Module):
    def __init__(self, feature_dim, num_sample=16, cylinder_radius=0.05, hmin=-0.02, hmax=0.04, num_angle=12, num_depth=4):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.in_dim = feature_dim
        mlps = [self.in_dim, 512, 256, 256]
        self.pcg = PointCylinderGroup(num_sample, mlps, cylinder_radius, hmin, hmax, bn=True)

        # (score + width) * num_angle * num_depth
        self.conv1 = nn.Conv1d(256, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.conv3 = nn.Conv1d(256, num_angle*num_depth*2, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, xyz, new_xyz, view_rot, features):
        B, num_seed, _ = new_xyz.size()
        view_features = self.pcg(xyz, new_xyz, view_rot, features)
        view_features = F.relu(self.bn1(self.conv1(view_features)), inplace=True)
        view_features = F.relu(self.bn2(self.conv2(view_features)), inplace=True)
        view_features = self.conv3(view_features)
        view_features = view_features.transpose(1,2).contiguous() #(B, Ns, A*D*2)
        return view_features

class MinkowskiGraspNet(nn.Module):
    def __init__(self, in_channels=3, num_seed=1024, num_view=300, num_angle=12, num_depth=4, sampling='heatmap_fps', view_selection='prob', is_training=True):
        super().__init__()
        self.num_seed = num_seed
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.view_selection = view_selection

        self.heatmap_generator = ResUNet14(in_channels=in_channels, out_channels=3, conv1_kernel_size=3, bn_momentum=0.02, D=3)
        self.view_estimator = ViewEstimator(in_channels=512, num_samples=num_seed, num_view=num_view, sampling=sampling, heatmap_th=0.1)
        self.grasp_generator = GraspGenerator(512, num_sample=16, cylinder_radius=0.05, hmin=-0.02, hmax=0.04, num_angle=num_angle, num_depth=num_depth)

    def forward(self, end_points):
        # generate heatmap
        sinput = end_points['sinput']
        soutput, sfeat = self.heatmap_generator(sinput, return_features=True)
        out_f = soutput.F
        end_points['stage1_objectness_pred'] = out_f[:,0:2]
        end_points['stage1_heatmap_pred'] = torch.sigmoid(out_f[:,2])
        end_points['stage1_point_features'] = sfeat

        # estimate grasp view
        obj_mask = torch.argmax(end_points['stage1_objectness_pred'], dim=1).bool()
        view_heatmap, seed_xyz, seed_inds, seed_features = self.view_estimator(sfeat, end_points['point_clouds'], obj_mask, end_points['stage1_heatmap_pred'])
        end_points['stage2_view_heatmap_pred'] = torch.sigmoid(view_heatmap)
        end_points['stage2_seed_xyz'] = seed_xyz
        end_points['stage2_seed_inds'] = seed_inds
        end_points['stage2_seed_features'] = seed_features

        # generate view proposal
        B = seed_xyz.size(0)
        # view_scores, view_inds = torch.max(end_points['stage2_view_heatmap_pred'], dim=2)
        if self.is_training:
            view_scores, view_inds = self.select_views(end_points, method=self.view_selection)
        else:
            view_scores, view_inds = self.select_views(end_points, method='top')
        view_inds_ = view_inds.view(B, self.num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous() #(B, num_seed, 1, 3)
        template_views = generate_grasp_views(self.num_view).to(view_heatmap.device) #(num_view, 3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, self.num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        view_xyz = torch.gather(template_views, 2, view_inds_).squeeze(2) #(B, num_seed, 3)
        view_xyz_ = view_xyz.view(B*self.num_seed, 3)
        batch_angle = torch.zeros(view_xyz_.size(0), dtype=view_xyz_.dtype, device=view_xyz_.device)
        view_rot = batch_viewpoint_params_to_matrix(-view_xyz_, batch_angle).view(B, self.num_seed, 3, 3)
        end_points['stage2_view_xyz'] = view_xyz
        end_points['stage2_view_inds'] = view_inds
        # end_points['stage2_view_rot'] = view_rot
        end_points['stage2_view_scores'] = view_scores

        # use grasp templates in training
        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_points = end_points['batch_grasp_point'].to(seed_xyz.device)
            approaching, end_points = match_grasp_view_and_label(end_points)
            approaching = approaching.to(seed_xyz.device)
        else:
            grasp_points = seed_xyz
            approaching = view_rot

        # generate grasp proposal
        view_features = self.grasp_generator(seed_xyz, grasp_points, approaching, seed_features)
        AD = self.num_angle * self.num_depth
        end_points['stage3_grasp_scores'] = torch.sigmoid(view_features[:,:,0:AD]).view(B, self.num_seed, self.num_angle, self.num_depth)
        end_points['stage3_normalized_grasp_widths'] = torch.sigmoid(view_features[:,:,AD:AD*2]).view(B, self.num_seed, self.num_angle, self.num_depth)

        return end_points

    def select_views(self, end_points, method):
        assert(method in ['prob', 'top'])
        view_preds = end_points['stage2_view_heatmap_pred']
        view_scores, view_inds = torch.max(view_preds, dim=2)
        if method == 'prob':
            # get view score masks
            view_inds_ = view_inds.unsqueeze(-1)
            top_mask = torch.zeros_like(view_preds)
            ones = torch.ones_like(view_preds)
            top_mask = top_mask.scatter(dim=2, index=view_inds_, src=ones)
            candidate_view_mask = ((view_preds>0.5) | top_mask.bool()).float()
            view_prob = view_preds * candidate_view_mask
            view_prob = view_prob.contiguous().view(-1, self.num_view)
            # select views
            view_inds = torch.multinomial(view_prob, num_samples=1)
            view_inds = view_inds.contiguous().view(-1, self.num_seed, 1)
            view_scores = torch.gather(view_preds, 2, view_inds).squeeze(-1)
            view_inds = view_inds.contiguous().view(-1, self.num_seed)
            end_points['stage2_candidate_views_count'] = candidate_view_mask.sum(dim=-1).mean()
        return view_scores, view_inds