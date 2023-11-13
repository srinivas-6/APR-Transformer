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
from PIL import Image, ImageFile

mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 11,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 5,  # pole
            18: 0,  # polegroup
            19: 6,  # traffic light
            20: 7,  # traffic sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23: 10,  # sky
            24: 0,  # person
            25: 0,  # rider
            26: 0,  # car
            27: 0,  # truck
            28: 0,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 0,  # motorcycle
            33: 0,  # bicycle
            -1: 0  # licenseplate
        }
mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (255, 0, 0),  # sidewalk
            9: (255, 0, 0),  # parking
            10: (255, 0, 0),  # rail track
            11: (255, 0, 0),  # building
            12: (255, 0, 0),  # wall
            13: (255, 0, 0),  # fence
            14: (255, 0, 0),  # guard rail
            15: (255, 0, 0),  # bridge
            16: (255, 0, 0),  # tunnel
            17: (255, 0, 0),  # pole
            18: (255, 0, 0),  # polegroup
            19: (255, 0, 0),  # traffic light
            20: (255, 0, 0),  # traffic sign
            21: (255, 0, 0),  # vegetation
            22: (255, 0, 0),  # terrain
            23: (0, 0, 255),  # sky
            24: (255, 0, 0),  # person
            25: (255, 0, 0),  # rider
            26: (255, 255, 0),  # car
            27: (255, 0, 0),  # truck
            28: (255, 0, 0),  # bus
            29: (255, 0, 0),  # caravan
            30: (255, 0, 0),  # trailer
            31: (255, 0, 0),  # train
            32: (255, 0, 0),  # motorcycle
            33: (255, 0, 0),  # bicycle
            -1: (255, 0, 0)  # licenseplate
        }

class APRSegDataset(Dataset):
    def __init__(self, cfg, mode='train', transforms=None, mask_transforms=None):
        self.mode = mode
        self.cfg = cfg
        self.root_dir = os.path.join(cfg['root_dir'], mode)
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.val_split = cfg['val_split']
        self.target_masks_dir = {'camera1_polygons': None,
                                 'camera2_polygons': None,
                                 'camera3_polygons': None,
                                 }
        self.num_classes = 12
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
    
    def get_masklist(self,path):
        _dir_ = os.path.join(self.root_dir, path)
        mask_list = []
        for f in os.listdir(_dir_):
            if f.endswith('.png'):
                mask_list.append(os.path.join(_dir_, f))
        return mask_list
    
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
        # load poses.txt file as metadata, drop NaN rows and target masks
        if self.mode == "val":
            self.root_dir = os.path.join(self.cfg['root_dir'], "train")
        with open(os.path.join(self.root_dir, 'poses.txt')) as f:
            metadata = pd.read_csv(f, sep=' ', header=None, names=['image1','image2','image3','x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'image1_masks', 'image2_masks', 'image3_masks'])
        # load target masks
        metadata['image1_masks'] = self.get_masklist('camera1_polygons')
        metadata['image2_masks'] = self.get_masklist('camera2_polygons')
        metadata['image3_masks'] = self.get_masklist('camera3_polygons')
        metadata = metadata.dropna() # drop NaN rows
        metadata = metadata.iloc[1:] # drop first row as it's a duplicate of column names          
        return metadata
    
    def get_pose(self,pose: list):
        return np.array(pose, dtype=np.float32)
    
    def get_mask(self,mask_file):
        # load mask in grayscale
        mask = imread(mask_file, as_gray=True)
        #mask = Image.open(mask_file).convert('L')
        return mask
    
    def mask_to_class(self, mask):
        '''
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mapping:
            maskimg[mask == k] = mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mappingrgb:
            rgbimg[0][mask == k] = mappingrgb[k][0]
            rgbimg[1][mask == k] = mappingrgb[k][1]
            rgbimg[2][mask == k] = mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in mapping.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = mappingrgb[mask2class[k]][2]
        return rgbimg
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        qpose = self.get_pose([row['x'], row['y'], row['z']])
        qrot =  self.get_pose([row['qw'], row['qx'], row['qy'], row['qz']])
        for i in self.cam_type:
            if i == 'camera1': # camera_front_wide
                img = self.get_image(os.path.join(self.root_dir, i , row['image1'])) 
                # next load the target
                target = self.get_mask(row['image1_masks'])
                qpose = self.get_pose(qpose) 
                qrot = self.get_pose(qrot)
            elif i =='camera2': # camera_front_right
                img = self.get_image(os.path.join(self.root_dir, i , row['image2'])) 
                qpose = self.get_pose(qpose) 
                qrot = self.get_pose(qrot)
                qrot = self.rotate_quaternion(qrot,math.radians(-60)) 
                target = self.get_mask(row['image2_masks'])
            elif i == 'camera3': # camera_front_left
                img = self.get_image(os.path.join(self.root_dir, i , row['image3'])) 
                qpose = self.get_pose(qpose) 
                qrot = self.get_pose(qrot)
                qrot = self.rotate_quaternion(qrot,math.radians(60)) 
                target = self.get_mask(row['image3_masks'])
            
            if self.transforms is not None:
                img = self.transforms(img)
                target = torch.from_numpy(np.array(target, dtype=np.uint8))
                targetmask = self.mask_to_class(target)
                targetmask = self.mask_transforms(targetmask)
            npose = self.normalize_pose(qpose, self.min_pose, self.max_pose) # normalize pose
            pose = np.concatenate((npose, qrot), axis=0) # concatenate pose and rotation
            
            sample = {
                'img': img, 
                'pose': torch.tensor(pose).float(),
                'mask': targetmask.long(),
                }
            
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



