import torch
import torch.nn as nn
from .voxelrcnn_head import VoxelRCNNHead
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable

class VoxelRCNNPoseHead(VoxelRCNNHead):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        
        super().__init__(backbone_channels, 
                         model_cfg, 
                         point_cloud_range, 
                         voxel_size, 
                         num_class=1, 
                         **kwargs)

        shared_dim = self.model_cfg.SHARED_FC[-1]
        latent_size = shared_dim
        self.fc1 = nn.Linear(shared_dim, latent_size)
        self.fc2 = nn.Linear(shared_dim, latent_size)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        # Distribution
        mu = self.fc1(shared_features)
        logvar = self.fc2(shared_features)
        z = self.reparametrize(mu, logvar)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(z))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['pred_mu'] = mu
            targets_dict['pred_logvar'] = logvar
            self.forward_ret_dict = targets_dict

        return batch_dict

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def get_loss(self, tb_dict=None):
        rcnn_loss, tb_dict = super().get_loss(tb_dict)
        # dist loss
        # mask
        rcnn_cls_labels = self.forward_ret_dict['rcnn_cls_labels'].view(-1)
        valid_mask = (rcnn_cls_labels > 0).long()

        pred_mu = self.forward_ret_dict['pred_mu'][valid_mask]
        pred_logvar = self.forward_ret_dict['pred_logvar'][valid_mask]
        pred_dist = Independent(Normal(loc=pred_mu, scale=torch.exp(pred_logvar)+3e-22), 1)
        normal_dist = Independent(Normal(loc=torch.zeros_like(pred_mu), scale=torch.ones_like(pred_logvar)), 1)
        normal_latent_loss = torch.mean(self.kl_divergence(pred_dist, normal_dist))
        rcnn_loss += normal_latent_loss
        normal_latent_dict = {
            'latent_loss': normal_latent_loss.item()
        }
        tb_dict.update(normal_latent_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict