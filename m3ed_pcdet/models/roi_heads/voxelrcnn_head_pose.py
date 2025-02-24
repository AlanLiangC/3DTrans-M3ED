import torch
import torch.nn as nn
from .voxelrcnn_head import VoxelRCNNHead
from torch.autograd import Variable
import torch.distributions as dist
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_inputs, probabilistic=True):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, 128)
        self.dropout = nn.Dropout(0.1)
        self.hiddens = nn.ModuleList([
            nn.Linear(128,128)
            for _ in range(2)])
        if probabilistic:
            self.output = nn.Linear(128, 64*2)
        else:
            self.output = nn.Linear(128, 64)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class VoxelRCNNPoseHead(VoxelRCNNHead):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        
        super().__init__(backbone_channels, 
                         model_cfg, 
                         point_cloud_range, 
                         voxel_size, 
                         num_class=1, 
                         **kwargs)

        shared_dim = self.model_cfg.SHARED_FC[-1]
        self.featurizer = MLP(shared_dim)
        self.classifier = nn.Linear(64, self.num_class, bias=True)

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
        z_params = self.featurizer(shared_features)
        z_dim = int(z_params.shape[-1]/2)
        z_mu, z_sigma = z_params[:,:z_dim], z_params[:,z_dim:]
        z_sigma = F.softplus(z_sigma) 
        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        z = z_dist.rsample()
        z_cls = self.classifier(z)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
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
            targets_dict['z_cls'] = z_cls

            self.forward_ret_dict = targets_dict

        return batch_dict

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def get_z_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['z_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels > 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels > 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'z_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        rcnn_loss, tb_dict = super().get_loss(tb_dict)

        z_loss_cls, cls_tb_dict = self.get_z_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += z_loss_cls
        tb_dict.update(cls_tb_dict)
        return rcnn_loss, tb_dict