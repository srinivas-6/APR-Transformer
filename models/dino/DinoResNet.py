import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import VisionTransformer

class DINOPoseNet(nn.Module):
    """
    A class to represent a classic pose regressor using DINO with an ResNet50 backbone
    """
    def __init__(self, config):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(DINOPoseNet, self).__init__()

        # DINO Resnet50
        # self.backbone = torch.hub.load('facebookresearch/dino:main', backbone_path)
        self.backbone =  VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        self.backbone.head = nn.Identity()
        self.backbone.load_state_dict(torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))
        dropout = config['dropout']
        backbone_dim = config['backbone_dim']
        latent_dim = config['latent_dim']
        self.freeze_backbone = config['freeze_backbone']

        # Regressor layers
        self.fc1 = nn.Linear(384, 384)
        self.fc2 = nn.Linear(384, 3)
        self.fc3 = nn.Linear(384, 4)

        self.dropout = nn.Dropout(p=dropout)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        x = self.backbone(data.get('img'))
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)
        return {'pose': torch.cat((p_x, p_q), dim=1)}

