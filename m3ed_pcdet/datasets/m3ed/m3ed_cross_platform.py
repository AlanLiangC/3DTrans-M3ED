import pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import copy
from ..dataset import DatasetTemplate
from . import m3ed_utils
from ...utils import box_utils

class M3ED_CP_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        '''
        M3ED dataset for cross platform
        '''
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        platform = dataset_cfg.PLATFORM_PATH[self.mode]
        self.root_split_path = self.root_path / platform / ('training' if self.mode != 'test' else 'testing')

        self.include_m3ed_data()

    def include_m3ed_data(self):
        anno_dict_path = self.root_split_path / 'labels.pkl'
        calib_dict_path = self.root_split_path / 'calibs.pkl'
        info_path = self.root_split_path.parent / 'trainvalSets' / ('training.txt' if self.mode != 'test' else 'testing.txt')
        with open(anno_dict_path, 'rb') as f:
            self.anno_dict = pickle.load(f)
        with open(calib_dict_path, 'rb') as f:
            self.calib_dict = pickle.load(f)
        self.image_shape = np.array([1280, 800])
        self.data_list_info = self.load_saved_frame_txt(info_path)
        if self.logger is not None:
            self.logger.info('Total samples for M3ED Cross Platcorm dataset: %d' % (len(self.calib_dict)))

    def __len__(self):
        return len(self.data_list_info)

    def load_saved_frame_txt(self, txt_path):

        save_frame_info = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.strip().split(' ')
                frame_info = dict(
                    platform = label[0],
                    terrian = label[1],
                    sequence = label[2],
                    frame_index = label[3]
                )
                save_frame_info.append(frame_info)
        f.close()
        return save_frame_info

    def get_lidar(self, frame_id):
        point_cloud_path = self.root_split_path / 'point_cloud' / f'{frame_id}.bin'
        points = np.fromfile(point_cloud_path, dtype=np.float64).reshape(-1, 4)
        return points
    
    def get_calib(self, frame_id):
        calib = self.calib_dict[frame_id]
        return calib

    def get_fov_flag(self, points, calib):
        extristric = calib['extristric']
        K = calib['K']
        D = calib['D']
        extend_points = m3ed_utils.cart_to_hom(points[:,:3])
        points_cam = extend_points @ extristric.T
        rvecs = np.zeros((3,1))
        tvecs = np.zeros((3,1))

        pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
                K, D)
        imgpts = pts_img[:,0,:]
        depth = points_cam[:,2]

        imgpts = np.round(imgpts)
        kept1 = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < self.image_shape[1]) & \
                    (imgpts[:, 0] >= 0) & (imgpts[:, 0] < self.image_shape[0]) & \
                    (depth > 0.0)
        return kept1

    def get_virtual_pose(self, pose, drone=False):
        # n --> virtual
        Ln_T_L0 = m3ed_utils.transform_inv(pose)
        r = Rotation.from_matrix(Ln_T_L0[:3,:3])
        Rn_T_R0_x, Rn_T_R0_y, _ = r.as_euler('xyz')
        R_x = m3ed_utils.rotx(Rn_T_R0_x)
        R_y = m3ed_utils.roty(Rn_T_R0_y)
        virtual_r_matrix = R_y @ R_x
        virtual_pose = np.eye(4)
        virtual_pose[:3, :3] = virtual_r_matrix

        if drone:
            position_n = Ln_T_L0[:3,3]
            change_z = position_n[2] - 1.7
            virtual_translate = np.eye(4)
            virtual_translate[2,3] = change_z
            virtual_pose = virtual_translate @ virtual_pose

        return virtual_pose

    def convert_boxes_from_vir_to_n(self, pose, boxes, drone=False):
        # vir pose
        virtual_pose = self.get_virtual_pose(pose, drone)
        pose = m3ed_utils.transform_inv(virtual_pose)
        if not drone:
            boxes[:,2] += 1.3
        r = Rotation.from_matrix(pose[:3,:3])
        ego2world_yaw = r.as_euler('xyz')[-1]
        boxes_global = boxes.copy()
        expand_centroids = np.concatenate([boxes[:, :3], 
                                           np.ones((boxes.shape[0], 1))], 
                                           axis=-1)
        centroids_global = np.dot(expand_centroids, pose.T)[:,:3]        
        boxes_global[:,:3] = centroids_global
        boxes_global[:,6] += ego2world_yaw
        return boxes_global

    def get_anno_info(self, index, calib):
        frame_id = str(index).zfill(5)
        anno_dict = copy.deepcopy(self.anno_dict[frame_id])
        frame_info = self.data_list_info[index]
        if frame_info['platform'] != 'car':
            converet_box = anno_dict['gt_boxes']
            if frame_info['platform'] == 'spot':
                drone = False
            else:
                drone = True
            if frame_info['sequence'] != 'srt_under_bridge_2':
                converet_box = self.convert_boxes_from_vir_to_n(calib['pose'], converet_box, drone)
            else:
                converet_box[:,2] += 1.3
            anno_dict['gt_boxes'] = converet_box
        return anno_dict

    def __getitem__(self, index):
        frame_id = str(index).zfill(5)
        points = self.get_lidar(frame_id)
        calib = self.get_calib(frame_id)

        if self.dataset_cfg.FOV_POINTS_ONLY:
            fov_flag = self.get_fov_flag(points, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': frame_id,
            'calib': calib,
            'image_shape': self.image_shape
        }
        anno_dict = self.get_anno_info(index, calib)
        input_dict.update({
            'gt_names': anno_dict['gt_names'],
            'gt_boxes': anno_dict['gt_boxes']
        })

        if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
            input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
            mask = np.zeros(anno_dict['gt_boxes'].shape[0], dtype=np.bool_)
            input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
            input_dict['gt_names'] = input_dict['gt_names'][mask]

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict