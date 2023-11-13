import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DataLoader
import cv2

class DeepLocDataset(Dataset):
    def __init__(self, cfg, mode='train', transforms=None):
        self.mode = mode
        self.cfg = cfg
        self.root_dir = os.path.join(cfg['root_dir'], mode)
        self.transforms = transforms
        self.val_split = cfg['val_split']
        self.metadata = self.load_metadata()
        self.min_pose = []
        self.max_pose = []
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
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_metadata(self):
        # load poses.txt file as metadata
        if self.mode == "val":
            self.root_dir = os.path.join(self.cfg['root_dir'], "train")
        with open(os.path.join(self.root_dir, 'poses.txt')) as f:
            metadata = pd.read_csv(f, sep=' ', header=None, names=['img_name', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            metadata = metadata.dropna() # drop NaN rows
            metadata = metadata.iloc[1:] # drop first row as it's a duplicate of column names           
        return metadata
    
    def get_pose(self,pose: list):
        return np.array(pose, dtype=np.float32)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        qpose = self.get_pose([row['x'], row['y'], row['z']])
        qrot =  self.get_pose([row['qw'], row['qx'], row['qy'], row['qz']])
        img = self.get_image(os.path.join(self.root_dir, 'LeftImages', row['img_name']+ '.png')) # load image
        qpose = self.get_pose(qpose) # load pose
        qrot = self.get_pose(qrot) # load rotation
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
    
    def qlog(self,q):
        """
        Applies logarithm map to q
        :param q: (4,)
        :return: (3,)
        """
        if all(q[1:] == 0):
            q = np.zeros(3)
        else:
            q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
        return q

    def qexp(q):
        """
        Applies the exponential map to q
        :param q: (3,)
        :return: (4,)
        """
        n = np.linalg.norm(q)
        q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
        return q

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.safe_load(f)
    return config

def test_dataloader():
    config = cfg_from_yaml_file('config/mapnet_deeploc.yaml')
    dataset = DeepLocDataset(config['DATASET'], mode='train', transforms=None)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['pose'].size())
        print(sample_batched['pose'])
        break


if __name__ == '__main__':
    test_dataloader()