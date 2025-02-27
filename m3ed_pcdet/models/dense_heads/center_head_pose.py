import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from .center_head import CenterHead

class PositionEncodingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.norm(x)
        x = self.activation(self.fc2(x))
        return x

class PoseNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(PoseNet, self).__init__()
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        prev_dim = input_dim
        for dim in hidden_dims:
            self.encoders.append(MLPBlock(prev_dim, dim))
            prev_dim = dim
        
        self.bottleneck = MLPBlock(hidden_dims[-1], hidden_dims[-1])
        
        for dim in reversed(hidden_dims[:-1]):
            self.decoders.append(MLPBlock(prev_dim * 2, dim))
            prev_dim = dim
        
        self.pose_estimitor = nn.Linear(hidden_dims[-1], 4)
        self.final_layer = nn.Linear(prev_dim * 2, input_dim)
    
    def forward(self, x):
        skip_connections = []
        
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        x = self.bottleneck(x)
        estimited_pose = self.pose_estimitor(x)
        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.final_layer(x)
        return estimited_pose, x

class CenterHead_Pose(CenterHead):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super(CenterHead_Pose, self).__init__(model_cfg, 
                                              input_channels, 
                                              num_class, 
                                              class_names, 
                                              grid_size, 
                                              point_cloud_range, 
                                              voxel_size,
                                              predict_boxes_when_training)
        
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE
        self.bev_pos = self.create_2D_grid(feature_map_size[0], feature_map_size[1])
        self.adp_max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.self_posembed = PositionEncodingLearned(input_channel=2, num_pos_feats=self.model_cfg.SHARED_CONV_CHANNEL)
        self.pose_est_model = PoseNet(self.model_cfg.SHARED_CONV_CHANNEL, hidden_dims=[128,64,32])

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def get_loss(self):
        loss, tb_dict = super().get_loss()
        # pose est
        pose_loss = F.l1_loss(self.forward_ret_dict['target_dicts']['pose_label'], self.forward_ret_dict['target_dicts']['pose_pred'])
        loss += pose_loss
        tb_dict['pose_loss'] = pose_loss.item()
        return loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        B,C,H,W = x.shape
        # [2, 64, 188, 94]
        # pose estimite
        # pooled_feature
        pooled_feature = self.adp_max_pool(x).view(B,C)
        estimited_pose, pose_latent_feature = self.pose_est_model(pooled_feature)
        pose_latent_feature = pose_latent_feature.view(B,C,1,1).repeat(1,1,H,W)
        bev_pos = self.bev_pos.repeat(B, 1, 1).to(pooled_feature.device)
        bev_position = self.self_posembed(bev_pos)
        bev_position = bev_position.reshape(B,C,W,H).permute(0,1,3,2)
        x = x + pose_latent_feature + bev_position
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            target_dict.update(
                {
                    'pose_label': torch.stack([torch.sin(data_dict['noise_rotation_x']),
                                                torch.cos(data_dict['noise_rotation_x']),
                                                torch.sin(data_dict['noise_rotation_y']),
                                                torch.cos(data_dict['noise_rotation_y'])
                                                ]).permute(1,0),
                    'pose_pred': estimited_pose
                }
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
