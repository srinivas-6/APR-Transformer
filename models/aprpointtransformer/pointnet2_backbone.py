import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

from time import sleep

class PointNetAbstraction(nn.Module):
    
    def __init__(self, point_feature_dim : int = 6, use_conv : bool = True, use_propagation : bool = True, dropout_rate : float = 0.2, checkpoint_path : str = None):
        super(PointNetAbstraction, self).__init__()

        # ...
        self.abstraction_dim = 128 if use_propagation else 256 # 1024 
        
        # ...
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], point_feature_dim, [[16, 16, 32], [32, 32, 64]]) # Was modified, 6 was originaly a 9;
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        
        # ...
        if use_propagation:
            self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
            self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
            self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
            # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            
        # ...
        if checkpoint_path is not None:
            self._load_model_checkpoint(checkpoint_path)
        
        # ...
        if use_conv:
            self.conv1 = nn.Conv1d(self.abstraction_dim, self.abstraction_dim, kernel_size=1)
            self.conv2 = nn.Conv1d(self.abstraction_dim, self.abstraction_dim, kernel_size=1)
        
        # ...
        self.bn = nn.BatchNorm1d(self.abstraction_dim)
        
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.use_propagation = use_propagation
        self.use_conv = use_conv
        
    def _load_model_checkpoint(self, checkpoint_path : str):
        
        # ...
        checkpoint_state_dict = torch.load(checkpoint_path)
            
        if 'model_state_dict' in checkpoint_state_dict.keys():
            checkpoint_state_dict = checkpoint_state_dict['model_state_dict']
            
        # ...
        model_state_dict = self.state_dict()
            
        # ...
        checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == checkpoint_state_dict[k].shape}
        model_state_dict.update(checkpoint_state_dict)
        
        # ...
        self.load_state_dict(model_state_dict)
            
            
    def forward(self, xyz):
    
        # ...
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        
        # ...
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # ...
        if self.use_propagation:
            l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            # l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # ...
        if self.use_conv:
            conv1_out = self.drop1(F.relu(self.bn(self.conv1(l1_points if self.use_propagation else l4_points))))
            conv2_out = self.drop2(self.conv2(conv1_out))

        # ...
        out = {}
        
        out["out_xyz"] = l1_xyz if self.use_propagation else l4_xyz
        out["out"] = conv2_out

        return out