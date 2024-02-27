import os
import torch
import numpy as np
import os.path as osp
from robotcar_dataset_sdk_pointloc.python.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from torch.utils import data
from PIL import Image
import torchvision
import transforms3d.quaternions as txq

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

# ...
class RadarRobotCar(data.Dataset):

    # ...
    def __init__(self, config, mode):

        # ...
        self.training = mode == 'train'
        self.data_dir = config['root_dir']
        self.scene = config['scene']
        self.divide_factor = config['divide_factor']

        # ...
        if self.training:

            seqs = ['2019-01-11-14-02-26-radar-oxford-10k',
                    '2019-01-14-12-05-52-radar-oxford-10k',
                    '2019-01-14-14-48-55-radar-oxford-10k',
                    '2019-01-18-15-20-12-radar-oxford-10k']
        
        elif not self.training:

            if self.scene=='full6':
                seqs=['2019-01-10-11-46-21-radar-oxford-10k'] # full 6

            elif self.scene=='full7':
                seqs=['2019-01-15-13-06-37-radar-oxford-10k'] # full 7

            elif self.scene=='full8':
                seqs=['2019-01-17-14-03-00-radar-oxford-10k'] # full 8

            elif self.scene=='full9':
                seqs=['2019-01-18-14-14-42-radar-oxford-10k'] # full 9

        # ...
        ps = {}
        ts = {}

        self.lidar_paths = []
        self.projected_lidar_paths = []

        all_poses_length = 0
        for seq in seqs:
            lidars_folder = osp.join(self.data_dir, seq, 'velodyne_left_fps_4096_3_float32_npy')
            lidars_list = os.listdir(lidars_folder)
            lidars_list = sorted(lidars_list)
            lidars_list = [
                int(lidar_name.replace('.npy','')) for lidar_name in lidars_list if lidar_name.endswith('.npy')
            ]
            ts_filename = osp.join(self.data_dir, seq, 'velodyne_left.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]
            assert ts[seq]==lidars_list
            ts[seq] = ts[seq][5:-5]
            lidars_list = lidars_list[5:-5]
            assert ts[seq]==lidars_list
            # poses from PointLoc
            pose_filename = osp.join(self.data_dir, seq, 'gps', 'ins.csv')
            p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq].copy(), ts[seq][0]))
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
            for lidar_name_pure in lidars_list:
                lidar_path = osp.join(lidars_folder, str(lidar_name_pure)+'.npy')
                self.lidar_paths.append(lidar_path) 

            all_poses_length += len(ps[seq])

        assert all_poses_length==len(self.lidar_paths)

        pose_stats_filename = osp.join(self.data_dir, 'pose_stats_full1234.txt')
        self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
        self.poses = np.empty((0, 7))
        self.pose_mean = self.mean_t
        self.pose_std = self.std_t
        for seq in seqs:
            pss = self.process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                              align_R=np.eye(3), align_t=np.zeros(3),
                              align_s=1)
            self.poses = np.vstack((self.poses, pss)) 

        # ...
        self.points_data = RawAPRPointCloudData(data_files = self.lidar_paths,  
                                                num_point = config['num_point'] if 'num_point' in config.keys() else 4096, 
                                                block_size = config['block_size'] if 'block_size' in config.keys() else 1000.0, 
                                                add_normalized_xyz = config['add_normalized_xyz'] if 'add_normalized_xyz' in config.keys() else True, 
                                                min_point_nr_per_frame = config['min_point_nr_per_frame'] if 'min_point_nr_per_frame' in config.keys() else 4096,
                                                down_sample_transform = config['down_sample_transform'] if 'down_sample_transform' in config.keys() else 'fps')


    def __len__(self):
        return len(self.poses)
    
    def process_poses(self, poses_in, mean_t, std_t, align_R, align_t, align_s):
        poses_out = np.zeros((len(poses_in), 7))
        poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
        # align
        for i in range(len(poses_out)):
            R = poses_in[i].reshape((3, 4))[:3, :3]
            q = txq.mat2quat(np.dot(align_R, R))
            q *= np.sign(q[0])  # constrain to hemisphere
            # q = self.qlog(q)
            poses_out[i, 3:] = q
            t = poses_out[i, :3] - align_t
            poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
        # normalize translation
        poses_out[:, :3] -= mean_t
        poses_out[:, :3] /= std_t
        return poses_out
    
    def __getitem__(self, index):

        lidar = self.points_data[index]
        lidar = np.swapaxes(lidar, 1, 0)
        lidar = torch.tensor(lidar,dtype=torch.float32)
        lidar = lidar / self.divide_factor

        pose = self.poses[index].copy()
        pose = torch.tensor(pose,dtype=torch.float32)

        return {'points':lidar, 'pose':pose}


# ...
class RawAPRPointCloudData(Dataset):
    
    def __init__(self, 
                 data_files : List[str] = None, data_root : str = None, ext : str = '.bin', 
                 num_point : int = 4096, block_size : float = 20.0, add_normalized_xyz : bool = True, min_point_nr_per_frame : int = 4096, 
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

            points = np.load(frame_path, allow_pickle=True) # Points Shape: N x 3 [x,y,z]
            
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
            
            elif point_idxs.size >= min_point_nr_per_frame:
                break

            else:
                increased_block_size = True
                self.block_size *= 1.1
         
        if increased_block_size:
            print(f"\nWarning: Increased block size to ~'{round(self.block_size, 4)}' to achive required number of points per point cloud frame.")

        # Downsample the given point cloud data:
        if point_idxs.size > self.num_point:
            
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
            
        elif point_idxs.size < self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        else:
            selected_point_idxs = point_idxs

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