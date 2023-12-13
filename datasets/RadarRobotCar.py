import os
import torch
import numpy as np
import os.path as osp
from robotcar_dataset_sdk_pointloc.python.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from torch.utils import data
from PIL import Image
import torchvision
import transforms3d.quaternions as txq


class RadarRobotCar(data.Dataset):
    def __init__(self, config, mode):
        self.training = mode == 'train'
        self.data_dir = config['root_dir']
        self.scene = config['scene']
        self.divide_factor = config['divide_factor']
        if self.training:
            seqs = [    
                # '2019-01-11-14-02-26-radar-oxford-10k',
                '2019-01-14-12-05-52-radar-oxford-10k',
                '2019-01-14-14-48-55-radar-oxford-10k',
                # '2019-01-18-15-20-12-radar-oxford-10k',
            ]
        
        elif not self.training:
            if self.scene=='full6':
                seqs=['2019-01-10-11-46-21-radar-oxford-10k'] # full 6

            elif self.scene=='full7':
                seqs=['2019-01-15-13-06-37-radar-oxford-10k'] # full 7

            elif self.scene=='full8':
                seqs=['2019-01-17-14-03-00-radar-oxford-10k'] # full 8

            elif self.scene=='full9':
                seqs=['2019-01-18-14-14-42-radar-oxford-10k'] # full 9
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
            projected_lidars_folder = osp.join(self.data_dir, seq, 'projected_lidar_64_720_shifted')
            for lidar_name_pure in lidars_list:
                projected_lidar_path = osp.join(projected_lidars_folder, str(lidar_name_pure)+'.png')
                self.projected_lidar_paths.append(projected_lidar_path)
            all_poses_length += len(ps[seq])
        assert all_poses_length==len(self.lidar_paths)
        assert all_poses_length==len(self.projected_lidar_paths)
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
    
    def lidar_to_histogram_features(self,lidar):
        """
        Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
        """
        def splat_points(point_cloud):
            # 256 x 256 grid
            pixels_per_meter = 8
            hist_max_per_pixel = 5
            x_meters_max = 30
            x_meters_min = -30
            y_meters_max = 50
            y_meters_min = -50
            xbins = np.linspace(x_meters_min, x_meters_max, 32*pixels_per_meter+1)
            ybins = np.linspace(y_meters_min, y_meters_max, 32*pixels_per_meter+1)
            hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
            hist[hist>hist_max_per_pixel] = hist_max_per_pixel
            overhead_splat = hist/hist_max_per_pixel
            return overhead_splat
        below = lidar[lidar[...,2]<=1.7]
        above = lidar[lidar[...,2]>1.7]
        below_features = splat_points(below)
        above_features = splat_points(above)
        features = np.stack([above_features, below_features], axis=-1).astype(np.float32)
        features = np.transpose(features, (2, 0, 1))
        features = np.rot90(features, 2, axes=(1,2)).copy()
        return features
    
    def __getitem__(self, index):
        lidar = np.load(self.lidar_paths[index],allow_pickle=True)
        projected_lidar = Image.open(self.projected_lidar_paths[index]).convert('RGB')
        pose = self.poses[index].copy()
        shuffle_ids = np.random.choice(len(lidar), size=len(lidar), replace=False)
        lidar = lidar[shuffle_ids]
        lidar_histogram = self.lidar_to_histogram_features(lidar)
        lidar = torch.tensor(lidar,dtype=torch.float32)
        lidar = lidar/self.divide_factor
        data_transform = []
        data_transform.append(torchvision.transforms.ToTensor())
        data_transform.append(torchvision.transforms.Normalize(mean=0.5, std=1))
        data_transform = torchvision.transforms.Compose(data_transform)
        projected_lidar = data_transform(projected_lidar)
        c,h,w = projected_lidar.shape
        projected_lidar = projected_lidar[:,h//2:,:]
        pose = torch.tensor(pose,dtype=torch.float32)
        return {
            'lidar_float32':lidar,
            'projected_lidar_float32':projected_lidar,
            'img':lidar_histogram,
            'image_float32':1,
            'bev_float32':1,
            'pose':pose,
        }