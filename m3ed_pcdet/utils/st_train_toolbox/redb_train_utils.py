import torch
import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist
from m3ed_pcdet.config import cfg
from m3ed_pcdet.models import load_data_to_gpu
from m3ed_pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
from m3ed_pcdet.models.model_utils.dsnorm import set_ds_target
from m3ed_pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from m3ed_pcdet.utils.box_utils import remove_points_in_boxes3d, enlarge_box3d
from m3ed_pcdet.ops.iou3d_nms import iou3d_nms_utils

import pickle as pkl
import re

#PSEUDO_LABELS = {}
from multiprocessing import Manager

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

PSEUDO_LABELS = Manager().dict() #for multiple GPU training
NEW_PSEUDO_LABELS = {}


def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            print ("********LOADING PS FROM:", cfg.SELF_TRAIN.INIT_PS)
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            return cur_pkl

    return None

def save_pseudo_label_epoch(model, val_loader, rank, leave_pbar, ps_label_dir, 
                            cur_epoch, source_reader=None, source_model=None):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)

    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    if cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    # related_boxes_count = Overlapped Boxes Counting (OBC) in paper
    related_boxes_count_list = [] if \
        cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None) else None

    # Since the model is eval status, some object-level data augmentation methods such as 
    # 'random_object_rotation', 'random_object_scaling', 'normalize_object_size' are not used 
    model.eval()

    total_quality_metric = None
    if cfg.SELF_TRAIN.get('REPORT_PS_LABEL_QUALITY', None) and \
            cfg.SELF_TRAIN.REPORT_PS_LABEL_QUALITY:
        total_quality_metric = {
            cls_id:{'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'scale_err': 0}
            for cls_id in range(len(cfg.CLASS_NAMES))}

    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

        pos_ps_batch, ign_ps_batch = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                         cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                         cur_epoch > 0),
            total_quality_metric=total_quality_metric,
            source_reader=source_reader,
            model=model,
            source_model=source_model,
            related_boxes_count_list=related_boxes_count_list
        )

        # log to console and tensorboard
        pos_ps_meter.update(pos_ps_batch)
        ign_ps_meter.update(ign_ps_batch)
        disp_dict = {'pos_ps_box': "{:.3f}({:.3f})".format(pos_ps_meter.val, pos_ps_meter.avg),
                     'ign_ps_box': "{:.3f}({:.3f})".format(ign_ps_meter.val, ign_ps_meter.avg)}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

        if cfg.SELF_TRAIN.get('PROGRESSIVE_SAMPLING', None) and cfg.SELF_TRAIN.PROGRESSIVE_SAMPLING.ENABLE and cur_epoch != cfg.OPTIMIZATION.NUM_EPOCHS:
            gt_reduce = cfg.SELF_TRAIN.PROGRESSIVE_SAMPLING.GT_REDUCE
            ps_grow = cfg.SELF_TRAIN.PROGRESSIVE_SAMPLING.PS_GROW
            if cfg.SELF_TRAIN.get('PS_SAMPLING', None):
                for k in cfg.SELF_TRAIN.PS_SAMPLING.SAMPLE_GROUPS:
                    cfg.SELF_TRAIN.PS_SAMPLING.SAMPLE_GROUPS[k] += ps_grow
            if cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NAME == 'gt_sampling' and \
                    'gt_sampling' not in cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.DISABLE_AUG_LIST:
                new_sample_groups = []
                for i in cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].SAMPLE_GROUPS:
                    new_sample_num = str(int(i.split(":")[-1])-gt_reduce)
                    new_i = i.split(":")[0] + ':' + new_sample_num
                    new_sample_groups.append(new_i)
                cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].SAMPLE_GROUPS = new_sample_groups

        if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
            # remove outliers
            related_boxes_count_list = [ i if len(i.shape) == 1
                                         else np.array([])
                                         for i in related_boxes_count_list]
            related_boxes_count_all = np.concatenate(related_boxes_count_list)

    model.train()

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)
    print(len(PSEUDO_LABELS))


def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True,
                            total_quality_metric=None,
                            source_reader=None,
                            model = None,
                            source_model=None,
                            related_boxes_count_list=None):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

            if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                before_nms_boxes = \
                    pred_dicts[b_idx]['pred_boxes_pre_nms'].detach().cpu().numpy()
            else:
                before_nms_boxes = None

            '''------------------ Cross-domain Examination (CDE) ------------------'''
            if cfg.SELF_TRAIN.get('CROSS_DOMAIN_DETECTION', None):
                pred_boxes_pnts = []
                batch_points = \
                    input_dict['points'][
                        input_dict['points'][:, 0] == b_idx][:,1:].cpu().numpy()
                internal_pnts_mask = \
                    points_in_boxes_cpu(batch_points, enlarge_box3d(pred_boxes[:, :7], extra_width=[1, 0.5, 0.5]))
                for msk_idx in range(pred_boxes.shape[0]):
                    pred_boxes_pnts.append(
                        batch_points[internal_pnts_mask[msk_idx] == 1])
                quality, selected, related_boxes_count = cross_domain_detection(
                    source_reader, pred_boxes, pred_scores, pred_labels,
                    pred_boxes_pnts, source_model,
                    input_dict['gt_boxes'][b_idx], before_nms_boxes, batch_points)

                quality_mask = np.zeros(len(pred_labels))
                quality_mask[selected] = True

                # remove boxes under negative threshold
                if cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.WITH_IOU_SCORE:

                    if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                        labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                        remain_mask = pred_scores >= labels_remove_scores
                        pred_labels = pred_labels[remain_mask]
                        pred_scores = pred_scores[remain_mask]
                        pred_boxes = pred_boxes[remain_mask]
                        if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                            related_boxes_count = related_boxes_count[remain_mask]
                        if 'pred_cls_scores' in pred_dicts[b_idx]:
                            pred_cls_scores = pred_cls_scores[remain_mask]
                        if 'pred_iou_scores' in pred_dicts[b_idx]:
                            pred_iou_scores = pred_iou_scores[remain_mask]

                    labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
                    ignore_mask = pred_scores < labels_ignore_scores
                    pred_labels[ignore_mask] = -1
                    
                if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
                    """ Count ralated boxes """
                    related_boxes_count_list.append(related_boxes_count[pred_labels==1])

                gt_box = np.concatenate((pred_boxes,
                                        pred_labels.reshape(-1, 1),
                                        pred_scores.reshape(-1, 1)), axis=1)
            else: # Not using CDE
                # remove boxes under negative threshold
                if cfg.SELF_TRAIN.get('NEG_THRESH', None):
                    labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                    remain_mask = pred_scores >= labels_remove_scores
                    pred_labels = pred_labels[remain_mask]
                    pred_scores = pred_scores[remain_mask]
                    pred_boxes = pred_boxes[remain_mask]
                    if 'pred_cls_scores' in pred_dicts[b_idx]:
                        pred_cls_scores = pred_cls_scores[remain_mask]
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_iou_scores[remain_mask]

                labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
                ignore_mask = pred_scores < labels_ignore_scores
                pred_labels[ignore_mask] = -pred_labels[ignore_mask]

                gt_box = np.concatenate((pred_boxes,
                                         pred_labels.reshape(-1, 1),
                                         pred_scores.reshape(-1, 1)), axis=1)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        '''--------- Target ReD Sampling ---------'''
        gt_points = None
        if cfg.SELF_TRAIN.get('PS_SAMPLING',None) and \
                cfg.SELF_TRAIN.PS_SAMPLING.ENABLE:
            gt_points = []
            batch_points = \
                input_dict['points'][
                    input_dict['points'][:,0]==b_idx][:,1:].cpu().numpy()
            internal_pnts_mask = \
                points_in_boxes_cpu(batch_points, gt_box[:, :7])
            for msk_idx in range(gt_box.shape[0]):
                gt_points.append(batch_points[internal_pnts_mask[msk_idx]==1])

        '''Ground Truth Infos for Saving & Next Round Training'''

        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'memory_counter': np.zeros(gt_box.shape[0])
        }
        if cfg.SELF_TRAIN.get('PS_SAMPLING',  None) and cfg.SELF_TRAIN.PS_SAMPLING.ENABLE:
            gt_infos.update({'gt_points': gt_points})
        if cfg.SELF_TRAIN.get('CROSS_DOMAIN_DETECTION', None) and cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.ENABLE:
            gt_infos.update({'quality_mask': quality_mask})
        if related_boxes_count_list is not None:
            gt_infos.update({'related_box_count': related_boxes_count if related_boxes_count_list is not None else None})
        # record pseudo label to pseudo label dict
        
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = ensemble_func(PSEUDO_LABELS[input_dict['frame_id'][b_idx]],
                                     gt_infos, cfg.SELF_TRAIN.MEMORY_ENSEMBLE)

        if gt_infos['gt_boxes'].shape[0] > 0:
            ign_ps_meter.update((gt_infos['gt_boxes'][:, 7] < 0).sum())
        else:
            ign_ps_meter.update(0)
        pos_ps_meter.update(gt_infos['gt_boxes'].shape[0] - ign_ps_meter.val)

        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos

    return pos_ps_meter.avg, ign_ps_meter.avg


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in PSEUDO_LABELS:
        gt_box = PSEUDO_LABELS[frame_id]['gt_boxes']
    else:
        raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)

    return gt_box

def count_tp_fp_fn_gt(pred_boxes, gt_boxes, iou_thresh=0.7, points=None):
    """ Count the number of tp, fp, fn and gt. Return tp boxes and their corresponding gt boxes
    """
    quality_metric = {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'scale_err': 0}
    assert gt_boxes.shape[1] == 7 and pred_boxes.shape[1] == 7
    quality_metric['gt'] += gt_boxes.shape[0]

    if gt_boxes.shape[0] == 0:
        quality_metric['fp'] += pred_boxes.shape[0]
        return None, None
    elif pred_boxes.shape[0] == 0:
        quality_metric['fn'] += gt_boxes.shape[0]
        return None, None

    pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)
    gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

    if not (pred_boxes.is_cuda and gt_boxes.is_cuda):
        pred_boxes, gt_boxes = pred_boxes.cuda(), gt_boxes.cuda()

    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7])
    max_ious, match_idx = torch.max(iou_matrix, dim=1)
    assert max_ious.shape[0] == pred_boxes.shape[0]

    # max iou > iou_thresh is tp
    tp_mask = max_ious >= iou_thresh
    ntps = tp_mask.sum().item()
    quality_metric['tp'] += ntps
    quality_metric['fp'] += max_ious.shape[0] - ntps

    # gt boxes that missed by tp boxes are fn boxes
    quality_metric['fn'] += gt_boxes.shape[0] - ntps

    # get tp boxes and their corresponding gt boxes
    tp_boxes = pred_boxes[tp_mask]
    tp_gt_boxes = gt_boxes[match_idx[tp_mask]]

    if ntps > 0:
        scale_diff, debug_boxes = cal_scale_diff(tp_boxes, tp_gt_boxes)
        quality_metric['scale_err'] += scale_diff

    return quality_metric, match_idx[tp_mask].cpu().numpy()

def cal_scale_diff(tp_boxes, gt_boxes):
    assert tp_boxes.shape[0] == gt_boxes.shape[0]

    aligned_tp_boxes = tp_boxes.detach().clone()

    # shift their center together
    aligned_tp_boxes[:, 0:3] = gt_boxes[:, 0:3]

    # align their angle
    aligned_tp_boxes[:, 6] = gt_boxes[:, 6]

    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(aligned_tp_boxes[:, 0:7], gt_boxes[:, 0:7])

    max_ious, _ = torch.max(iou_matrix, dim=1)

    scale_diff = (1 - max_ious).sum().item()

    return scale_diff, aligned_tp_boxes.cpu().numpy()

def cross_domain_detection(source_reader, pred_boxes, pred_scores, pred_labels,
                           pred_boxes_pnts, model, target_gt_box, before_nms_boxes=None,
                           target_points=None):

    if cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.ENABLE:
        b_idx = 0
        source_batch = source_reader.read_data()
        # get all points of a single PC from batch [# of point, 3]
        single_pc_pnts = \
            source_batch['points'][source_batch['points'][:, 0] == b_idx][:, 1:]
        s_gt_box = source_batch['gt_boxes'][b_idx] # source gt box

        s_gt_box = enlarge_box3d(s_gt_box, extra_width=[1, 0.5, 0.5])
        # Remove GT boxes and points in source PC [# of point, 3]
        single_pc_pnts = \
            remove_points_in_boxes3d(single_pc_pnts, s_gt_box[:,:7])

        # Remove points at PS boxes in source PC [# of point, 3]
        single_pc_pnts = \
            remove_points_in_boxes3d(single_pc_pnts, enlarge_box3d(pred_boxes[:, :7], extra_width=[1, 0.5, 0.5]))


        # remove all points of this single PC from batch [# of point, 4]
        source_batch['points'] = \
            source_batch['points'][source_batch['points'][:, 0] != b_idx]

        # Add PS objects points into source PC
        ps_pnts_to_sample = None
        for obj_pnts in pred_boxes_pnts:
            ps_pnts_to_sample = obj_pnts if ps_pnts_to_sample is None else np.concatenate([ps_pnts_to_sample, obj_pnts])
        try:
            single_pc_pnts = np.concatenate([ps_pnts_to_sample, single_pc_pnts])
        except ValueError:
            pass # ps_pnts_to_sample is None

        """  Rebuild voxels to batch"""
        config = cfg.DATA_CONFIG.DATA_PROCESSOR[-1]
        if config.get('VOXEL_SIZE', None):
            voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                num_point_features=source_reader.dataloader.dataset.point_feature_encoder.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[source_reader.dataloader.dataset.mode]
            )

            voxel_output = voxel_generator.generate(single_pc_pnts)
            voxels, coordinates, num_points = voxel_output

            voxel_coords_single_pc = source_batch['voxel_coords'][
                source_batch['voxel_coords'][:, 0] == b_idx]
            source_batch['voxel_coords'] = source_batch['voxel_coords'][
                source_batch['voxel_coords'][:, 0] != b_idx]
            voxel_num = len(voxel_coords_single_pc)

            # Combine processed voxels to existing voxels,
            # remove existed voxels in this batch as well
            source_batch['voxels'] = \
                np.concatenate((voxels, source_batch['voxels'][voxel_num:]))
            source_batch['voxel_num_points'] = \
                np.concatenate((num_points,
                                source_batch['voxel_num_points'][voxel_num:]))
            batch_dim_to_cat = np.zeros(voxels.shape[0])
            batch_dim_to_cat[batch_dim_to_cat==0] = b_idx
            coordinates = np.concatenate(
                [batch_dim_to_cat.reshape(coordinates.shape[0], 1), coordinates],
                axis=1)
            source_batch['voxel_coords'] = \
                np.concatenate((coordinates, source_batch['voxel_coords']))

        """  Rebuild points to batch  """
        batch_dim_to_cat = np.zeros(single_pc_pnts.shape[0])
        batch_dim_to_cat[batch_dim_to_cat==0] = b_idx
        single_pc_pnts = np.concatenate(
            [batch_dim_to_cat.reshape(single_pc_pnts.shape[0], 1), single_pc_pnts],
            axis=1)
        source_batch['points'] = \
            np.concatenate([single_pc_pnts, source_batch['points']])

        load_data_to_gpu(source_batch)
        batch_pred_dict = model(source_batch)[0][b_idx]
        pred_boxes_from_source = batch_pred_dict['pred_boxes']
        pred_labels_from_source = batch_pred_dict['pred_labels']

        quality_metric, selected_ids = count_tp_fp_fn_gt(
            pred_boxes_from_source[:, 0:7],
            pred_boxes[:, 0:7],
            iou_thresh=cfg.SELF_TRAIN.CROSS_DOMAIN_DETECTION.CDE_IOU_TH
        )

    else:
        quality_metric=None
        selected_ids=np.array(range(0, pred_boxes.shape[0]), dtype='i')

    """ Counting OBC before NMS to each predicted boxes """
    pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)
    related_boxes_count = 0
    if cfg.SELF_TRAIN.get('DIVERSITY_SAMPLING', None):
        before_nms_boxes, _ = common_utils.check_numpy_to_torch(before_nms_boxes)
        if not (pred_boxes.is_cuda and before_nms_boxes.is_cuda):
            pred_boxes, before_nms_boxes = pred_boxes.cuda(), \
                                           before_nms_boxes.cuda()

        iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(pred_boxes[:, :7].cpu().numpy(),
                                          before_nms_boxes.cpu().numpy())

        related_boxes = iou_matrix > 0.3
        related_boxes_count = (related_boxes != 0).sum(axis=1) # OBC

    return quality_metric, selected_ids, related_boxes_count

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points