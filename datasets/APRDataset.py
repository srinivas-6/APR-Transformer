import os
import time
from typing import List, Literal

from tqdm import tqdm

import yaml
import pandas as pd

import math
import numpy as np
import fpsample
from scipy.spatial.transform import Rotation as R

import cv2
from skimage.io import imread

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DataLoader

class APRDataset(Dataset):
    
    # ...
    def __init__(self, cfg, mode='train', img_transforms=None, points_transforms=None):
        
        # ...
        self.mode = mode
        self.cfg = cfg
        
        self.root_dir = os.path.join(cfg['root_dir'], mode)
        
        self.val_split = cfg['val_split']
        
        # ...
        self.img_transforms = img_transforms
        self.points_transforms = points_transforms
        
        # ...
        self.metadata = self.load_metadata()
        
        # ...
        self.pre_process = cfg['pre-process'] if 'pre-process' in cfg.keys() else True

        # ...
        self.modality = cfg['modality']
        
        if self.modality == 'image':
            self.cam_type = cfg['cam_type']
        
        elif self.modality == 'lidar':
            self.lidar_type = cfg['lidar_features'] if 'lidar_features' in cfg.keys() else 'histogram'
            
        else:
            raise NotImplementedError(f'Modality "{self.modality}" is not implemented yet.')
        
        # ...
        if mode == 'train':
            self.metadata = self.metadata[:int(len(self.metadata) * (1 - self.val_split))]
            
        elif mode == 'val':
            self.metadata = self.metadata[int(len(self.metadata) * (1 - self.val_split)):]
            
        else:
            pass
        
        # ...
        if self.modality == 'lidar' and self.lidar_type == 'points':
            
            # ...
            data_files = []
            for idx in range(len(self.metadata)):
                
                data_file = os.path.join(self.root_dir, 'lidar', self.metadata.iloc[idx]['lidar'])
                data_files.append(data_file)
            
            # ...
            points_data = RawAPRPointCloudData(data_files = data_files,  
                                               num_point = cfg['num_point'] if 'num_point' in cfg.keys() else 4096, 
                                               block_size = cfg['block_size'] if 'block_size' in cfg.keys() else 20.0, 
                                               add_normalized_xyz = cfg['add_normalized_xyz'] if 'add_normalized_xyz' in cfg.keys() else True, 
                                               min_point_nr_per_frame = cfg['min_point_nr_per_frame'] if 'min_point_nr_per_frame' in cfg.keys() else 1024,
                                               down_sample_transform = cfg['down_sample_transform'] if 'down_sample_transform' in cfg.keys() else 'fps')
            
            # ...
            if self.pre_process:
                
                self.points = np.zeros((
                                    len(data_files), 
                                    cfg['num_point'] if 'num_point' in cfg.keys() else 4096, 
                                    6 if (cfg['add_normalized_xyz'] if 'add_normalized_xyz' in cfg.keys() else True) else 3
                                    ))
                
                points_dataloader = DataLoader(points_data, 
                                               num_workers=cfg['n_pre-process_workers'] if 'n_pre-process_workers' in cfg.keys() else 0, 
                                               worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
        
                for idx, point_data in enumerate(tqdm(points_dataloader)):
                    self.points[idx] = point_data
                
            else:
                self.points = points_data
            
        # ...
        self.min_pose, self.max_pose = [], []
        self.get_min_max_pose()
        
    
    # ...
    def get_min_max_pose(self):
        
        if self.mode != 'test':
            
            poses = []
            for i in range(0, len(self.metadata)):
                row = self.metadata.iloc[i]
                qpose = self.get_pose([row['x'], row['y'], row['z']])
                poses.append(qpose)
                
            poses = np.array(poses)
            self.min_pose = np.min(poses, axis=0)
            self.max_pose = np.max(poses, axis=0)
            
            # Create a df and write as txt:
            df = pd.DataFrame([self.min_pose, self.max_pose])
            df.to_csv(os.path.join(self.cfg['root_dir'], 'pose_meta.txt'), index=False, header=False)
            
        else:
            print(f'Loading pose meta as mode is {self.mode}')
            self.min_pose, self.max_pose = self.load_pose_meta(os.path.join(self.cfg['root_dir'], 'pose_meta.txt'))

    # ...
    def normalize_pose(self, pose, min_pose, max_pose):
        
        if len(min_pose) == 0:
            self.get_min_max_pose()
            
        pose = (pose - min_pose) / (max_pose - min_pose)
        return pose
    
    # ...
    def to_torch_tensor(self, img):
        return torch.from_numpy(img.transpose((2, 0, 1))).float()
    
    # ...
    def rotate_quaternion(self, quaternion, angle):
        
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        rotation_matrix = rotation_matrix @ R.from_euler('z', angle).as_matrix()
        
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        return quaternion
            
    # ...
    def load_pose_meta(self, meta_file):
        
        # Load as df and convert to numpy array:
        df = pd.read_csv(meta_file, header=None)
        
        min_pose = df.iloc[0].values
        max_pose = df.iloc[1].values
        
        return min_pose, max_pose
    
    # ...
    def get_image(self, img_file):
        
        img = imread(img_file)
        return img
    
    # ...
    def lidar_to_histogram_features(self,lidar):
        
        """
        Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
        """
        def splat_points(point_cloud):
            
            # Parameters for a 256 x 256 grid:
            pixels_per_meter = 8
            hist_max_per_pixel = 5
            
            x_meters_max = 30
            x_meters_min = -30
            y_meters_max = 40
            y_meters_min = -10
            
            # ...
            xbins = np.linspace(x_meters_min, x_meters_max, 32*pixels_per_meter+1)
            ybins = np.linspace(y_meters_min, y_meters_max, 32*pixels_per_meter+1)
            
            hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
            hist[hist>hist_max_per_pixel] = hist_max_per_pixel
            
            overhead_splat = hist/hist_max_per_pixel
            return overhead_splat
        
        # ...
        below = lidar[lidar[...,2]<=2.0]
        below_features = splat_points(below)
        
        above = lidar[lidar[...,2]>2.0]
        above_features = splat_points(above)
        
        features = np.stack([above_features, below_features], axis=-1).astype(np.float32)
        features = np.transpose(features, (2, 0, 1))
        
        return features
    
    # ...
    def load_metadata(self):
        
        # load poses.txt file as metadata
        if self.mode == "val":
            self.root_dir = os.path.join(self.cfg['root_dir'], "train")
            
        with open(os.path.join(self.root_dir, 'poses.txt')) as f:
            
            metadata = pd.read_csv(f, sep=' ', header=None, names=['image1','image2','image3','lidar','x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            metadata = metadata.dropna() # drop NaN rows
            metadata = metadata.iloc[1:] # drop first row as it's a duplicate of column names    
            
        return metadata
    
    # ...
    def get_pose(self,pose: list):
        return np.array(pose, dtype=np.float32)
    
    # ...
    def load_image_modality_data(self, row, cam_type, qpose, qrot):
        
        if cam_type == 'camera1': # camera_front_wide
                    
            img = self.get_image(os.path.join(self.root_dir, cam_type , row['image1'])) 
            
            qpose = self.get_pose(qpose) 
            qrot = self.get_pose(qrot)
                
        elif cam_type =='camera2': # camera_front_right
            
            img = self.get_image(os.path.join(self.root_dir, cam_type , row['image2'])) 
                    
            qpose = self.get_pose(qpose) 
            qrot = self.get_pose(qrot)
            qrot = self.rotate_quaternion(qrot,math.radians(-60)) 
            
        elif cam_type == 'camera3': # camera_front_left
            
            img = self.get_image(os.path.join(self.root_dir, cam_type , row['image3'])) 
                    
            qpose = self.get_pose(qpose) 
            qrot = self.get_pose(qrot)
            qrot = self.rotate_quaternion(qrot,math.radians(60)) 
            
        if self.img_transforms is not None:
            
            img = self.img_transforms(img)
                
        npose = self.normalize_pose(qpose, self.min_pose, self.max_pose) # normalize pose
        pose = np.concatenate((npose, qrot), axis=0) # concatenate pose and rotation
        
        return img, pose
    
    # ...
    def load_lidar_modality_data(self, lidar_type, qpose, qrot, idx : int = None, row = None): 
        
        # ...
        if lidar_type == 'histogram':
    
            # ...
            points = np.fromfile(os.path.join(self.root_dir, 'lidar', row['lidar']), dtype=np.float32).reshape(-1, 4)
            points_histogram = self.lidar_to_histogram_features(points)
            
            if self.points_transforms is not None:
                points = self.points_transforms(points)
                
            # ...
            qpose = self.get_pose(qpose) 
            qrot = self.get_pose(qrot)
            npose = self.normalize_pose(qpose, self.min_pose, self.max_pose) # normalize pose
            pose = np.concatenate((npose, qrot), axis=0) # concatenate normalized pose and rotation
            
            sample = {'img': torch.tensor(points_histogram).float(), 'pose': torch.tensor(pose).float()}
            return sample
        
        # ...
        elif lidar_type == 'points':
    
            # ...
            points = self.points[idx]
        
            if self.points_transforms is not None:
                points = self.points_transforms(points)
                
            points = np.swapaxes(points, 1, 0)
    
            # ...
            qpose = self.get_pose(qpose) 
            qrot = self.get_pose(qrot)
            npose = self.normalize_pose(qpose, self.min_pose, self.max_pose) # normalize pose
            pose = np.concatenate((npose, qrot), axis=0) # concatenate normalized pose and rotation
            
            sample = {'points': torch.tensor(points).float(), 'pose': torch.tensor(pose).float()}
            return sample   
    
    # ...
    def __getitem__(self, index):
        
        # ...
        row = self.metadata.iloc[index]
        
        qpose = self.get_pose([row['x'], row['y'], row['z']])
        qrot =  self.get_pose([row['qw'], row['qx'], row['qy'], row['qz']])
        
        # ...
        if self.modality == 'image':
            for cam_type in self.cam_type:

                img, pose = self.load_image_modality_data(row, cam_type, qpose, qrot)
                sample = {'img': img, 'pose': torch.tensor(pose).float()}
                
                return sample                                                              # TODO: Check, if 'sample' should be returned here;
            
        # ...
        elif self.modality == 'lidar':
            
            sample = self.load_lidar_modality_data(self.lidar_type, qpose, qrot, idx = index, row = row) 
            return sample
        
        else:
            raise NotImplementedError(f'Modality "{self.modality}" is not implemented yet.')

    # ...
    def __len__(self):
        return len(self.metadata)


# ...
class RawAPRPointCloudData(Dataset):
    
    def __init__(self, 
                 data_files : List[str] = None, data_root : str = None, ext : str = '.bin', 
                 num_point : int = 4096, block_size : float = 20.0, add_normalized_xyz : bool = True, min_point_nr_per_frame : int = 1024, 
                 transform = None, down_sample_transform : Literal["fps", "fps_approx", "random"] = "fps",
                 numpy_seed : int = None, torch_seed : int = None):
        
        # ...
        if data_files is None and data_root is None:
            raise ValueError("Either a list of file paths ('data_files') or a data root directory ('data_root') needs to be specified.")
        
        if data_files is not None and data_root is not None:
            raise ValueError("You should only specify either 'data_files' or 'data_root'.")
        
        # ...
        if numpy_seed is not None:
            np.random.seed(numpy_seed)
        
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        
        # ...
        super().__init__()
        
        # ...
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.down_sample_transform = down_sample_transform
        
        self.add_normalized_xyz = add_normalized_xyz
        
        self.min_point_nr_per_frame = min_point_nr_per_frame
        
        # Collect all 'Point Cloud Frame'-Files in the given data directory (ending with specified extension):
        point_cloud_frame_file_names = data_files if data_files is not None else [f for f in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, f)) and f.endswith(ext if ext is not None else "")] 
        self.point_cloud_frame_file_names = point_cloud_frame_file_names
        
        # ...
        self.points = []
        self.coord_min, self.coord_max = [], []
        
        # Load Point Cloud data from frame files (+ compute min & max point coordinates for each frame):
        num_point_all = []
        for frame_file_name in tqdm(point_cloud_frame_file_names, total=len(point_cloud_frame_file_names)):
            
            frame_path = frame_file_name if data_files is not None else os.path.join(data_root, frame_file_name)
            points = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4)[:, :3] # Points Shape: N x 3 [x,y,z]
            
            coord_min, coord_max = np.amin(points, axis=0), np.amax(points, axis=0)
            
            self.points.append(points)
            self.coord_min.append(coord_min), self.coord_max.append(coord_max)
            
            num_point_all.append(points.shape[0])
            
        # ...
        frame_idxs = []
        for index in range(len(point_cloud_frame_file_names)):
            frame_idxs.append(index)
            
        self.frame_idxs = np.array(frame_idxs)
        
        
    def __getitem__(self, idx):
        
        # ...
        frame_idx = self.frame_idxs[idx]  # <- K
        points = self.points[frame_idx]   # <- N x 3
        
        N_points = points.shape[0]        # <- N
        
        # ...
        min_point_nr_per_frame  = None if N_points < self.min_point_nr_per_frame else self.min_point_nr_per_frame 
        
        # ...
        center = np.array([0, 0, 0])
        
        increased_block_size = False
        while (True):

            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            
            if min_point_nr_per_frame is None:
                break
            
            elif point_idxs.size > min_point_nr_per_frame:
                break

            else:
                increased_block_size = True
                self.block_size *= 1.1
         
        if increased_block_size:
            print(f"\nWarning: Increased block size to ~'{round(self.block_size, 4)}' to achive required number of points per point cloud frame.")

        # Downsample the given point cloud data:
        if point_idxs.size >= self.num_point:
            
            if self.down_sample_transform == "fps":
                selected_point_idxs = point_idxs[fpsample.fps_sampling(points[point_idxs], self.num_point)]
                
            elif self.down_sample_transform == "fps_approx":
                selected_point_idxs = point_idxs[fpsample.fps_npdu_kdtree_sampling(points[point_idxs], self.num_point)]

            elif self.down_sample_transform == "kdtree":
                selected_point_idxs = point_idxs[fpsample.bucket_fps_kdtree_sampling(points[point_idxs], self.num_point)]

            elif self.down_sample_transform == "random":
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            
            else:
                raise NotImplementedError(f"Selected downsampling method '{self.down_sample_transorm}' isn't implemented.")
            
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # ...
        selected_points = points[selected_point_idxs, :]  # num_point x 3
        
        if self.add_normalized_xyz:
            
            current_points = np.zeros((self.num_point, 6))  # num_point x 6
        
            current_points[:, 3] = selected_points[:, 0] / self.coord_max[frame_idx][0]
            current_points[:, 4] = selected_points[:, 1] / self.coord_max[frame_idx][1]
            current_points[:, 5] = selected_points[:, 2] / self.coord_max[frame_idx][2]

            current_points[:, 0:3] = selected_points

        else:
            current_points = selected_points
            
        # ...
        if self.transform is not None:
            current_points = self.transform(current_points)
            
        return current_points

    def __len__(self):
        return len(self.frame_idxs)
