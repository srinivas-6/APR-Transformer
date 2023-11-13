import os
import yaml
from skimage.io import imread
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DataLoader
import cv2
import math
from scipy.spatial.transform import Rotation as R

class APRDataset(Dataset):
    def __init__(self, cfg, mode='train', transforms=None):
        self.mode = mode
        self.cfg = cfg
        self.root_dir = os.path.join(cfg['root_dir'], mode)
        self.transforms = transforms
        self.val_split = cfg['val_split']
        self.metadata = self.load_metadata()
        self.min_pose = []
        self.max_pose = []
        self.cam_type = cfg['cam_type']
        self.get_min_max_pose()
        if mode == 'train':
            self.metadata = self.metadata[:int(len(self.metadata) * (1 - self.val_split))]
        elif mode == 'val':
            self.metadata = self.metadata[int(len(self.metadata) * (1 - self.val_split)):]
        else:
            pass
        
    def __len__(self):
        return len(self.metadata)
    
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
            # create a df and write as txt
            df = pd.DataFrame([self.min_pose, self.max_pose])
            df.to_csv(os.path.join(self.cfg['root_dir'], 'pose_meta.txt'), index=False, header=False)
        else:
            print(f'Loading pose meta as mode is {self.mode}')
            self.min_pose, self.max_pose = self.load_pose_meta(os.path.join(self.cfg['root_dir'], 'pose_meta.txt'))

    def load_pose_meta(self,meta_file):
       # load as df and convert to numpy array
        df = pd.read_csv(meta_file, header=None)
        min_pose = df.iloc[0].values
        max_pose = df.iloc[1].values
        return min_pose, max_pose
    
    def get_image(self, img_file):
        img = imread(img_file)
        # img = cv2.imread(img_file)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_metadata(self):
        # load poses.txt file as metadata
        if self.mode == "val":
            self.root_dir = os.path.join(self.cfg['root_dir'], "train")
        with open(os.path.join(self.root_dir, 'poses.txt')) as f:
            metadata = pd.read_csv(f, sep=' ', header=None, names=['image1','image2','image3','x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            metadata = metadata.dropna() # drop NaN rows
            metadata = metadata.iloc[1:] # drop first row as it's a duplicate of column names           
        return metadata
    
    def get_pose(self,pose: list):
        return np.array(pose, dtype=np.float32)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        qpose = self.get_pose([row['x'], row['y'], row['z']])
        qrot =  self.get_pose([row['qw'], row['qx'], row['qy'], row['qz']])
        for i in self.cam_type:
            if i == 'camera1': # camera_front_wide
                img = self.get_image(os.path.join(self.root_dir, i , row['image1'])) 
                qpose = self.get_pose(qpose) 
                qrot = self.get_pose(qrot)
            elif i =='camera2': # camera_front_right
                img = self.get_image(os.path.join(self.root_dir, i , row['image2'])) 
                qpose = self.get_pose(qpose) 
                qrot = self.get_pose(qrot)
                qrot = self.rotate_quaternion(qrot,math.radians(-60)) 
            elif i == 'camera3': # camera_front_left
                img = self.get_image(os.path.join(self.root_dir, i , row['image3'])) 
                qpose = self.get_pose(qpose) 
                qrot = self.get_pose(qrot)
                qrot = self.rotate_quaternion(qrot,math.radians(60)) 
            
            if self.transforms is not None:
                img = self.transforms(img)

            npose = self.normalize_pose(qpose, self.min_pose, self.max_pose) # normalize pose
            pose = np.concatenate((npose, qrot), axis=0) # concatenate pose and rotation
            
            sample = {
                'img': img, 
                'pose': torch.tensor(pose).float()}
            
            return sample

    def normalize_pose(self,pose, min_pose, max_pose):
        if len(min_pose) == 0:
            self.get_min_max_pose()
        pose = (pose - min_pose) / (max_pose - min_pose)
        return pose
    
    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1))).float()
    
    def rotate_quaternion(self,quaternion, angle):
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        rotation_matrix = rotation_matrix @ R.from_euler('z', angle).as_matrix()
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        return quaternion


