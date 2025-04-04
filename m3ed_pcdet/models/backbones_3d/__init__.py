from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, VoxelWideResBackBone8x, VoxelWideResBackBone_L8x
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
from .spconv_backbone_unibn import VoxelBackBone8x_UniBN, VoxelResBackBone8x_UniBN
from .spconv_unet import UNetV2
from .IASSD_backbone import IASSD_Backbone
from .hednet import HEDNet, HEDNet2D
from .hednet import SparseHEDNet, SparseHEDNet2D
from .SSD_backbone import SSDBackbone

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelWideResBackBone8x': VoxelWideResBackBone8x,
    'VoxelWideResBackBone_L8x': VoxelWideResBackBone_L8x,
    # Dataset-specific Norm Layer
    'VoxelBackBone8x_UniBN':VoxelBackBone8x_UniBN,
    'VoxelResBackBone8x_UniBN':VoxelResBackBone8x_UniBN,
    'IASSD_Backbone': IASSD_Backbone,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'HEDNet': HEDNet,
    'HEDNet2D': HEDNet2D,
    'SparseHEDNet': SparseHEDNet,
    'SparseHEDNet2D': SparseHEDNet2D,
    'SSDBackbone': SSDBackbone
}
