"""
Backbone code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- use efficient-net as backbone and extract different activation maps from different reduction maps
- change learned encoding to have a learned token for the pose
"""

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

from .pointnet2_backbone import PointNetAbstraction

from .pencoder import build_position_encoding, NestedTensor


# ...
class EfficientNetBackboneBase(nn.Module):
    
    def __init__(self, backbone: nn.Module, reduction): 
        super().__init__()
        
        # ...
        self.body = backbone
        self.reductions = reduction
        self.reduction_map = {"reduction_3": 40, "reduction_4": 112}
        self.num_channels = [self.reduction_map[reduction] for reduction in self.reductions]

    def forward(self, tensor_list: NestedTensor):
        
        # ...
        xs = self.body.extract_endpoints(tensor_list.tensors) 
        out: Dict[str, NestedTensor] = {}
        
        # ...
        for name in self.reductions:
            
            x = xs[name]
            m = tensor_list.mask
            
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            
            out[name] = NestedTensor(x, mask)
            
        return out


class EfficientNetBackbone(EfficientNetBackboneBase):
    
    def __init__(self, pre_backbone: str, reduction):
        
        backbone = EfficientNet.from_pretrained(pre_backbone)
        backbone._conv_stem.in_channels = 2
        backbone._conv_stem.weight = torch.nn.Parameter(backbone._conv_stem.weight[:, :2, :, :])
        
        super().__init__(backbone, reduction)

        
# ...
class PointNetAbstractionBackboneBase(nn.Module):
    
    def __init__(self, backbone: nn.Module, reduction: list, reduction_map: dict): 
        super().__init__()
        
        # ...
        self.body = backbone
        self.reductions = reduction
        self.reduction_map = reduction_map
        self.num_channels = [self.reduction_map[reduction] for reduction in self.reductions]

    def forward(self, tensor_list: NestedTensor):
        
        # ...
        xs = self.body(tensor_list.tensors) 
        out: Dict[str, NestedTensor] = {}
        
        # ...
        for idx, name in enumerate(self.reductions):
            
            x = xs[name]
            x_pos = xs[name + "_xyz"]
            
            m = tensor_list.mask
            
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[2:]).to(torch.bool)[0]
            
            out[f"{name}_{idx}"] = [NestedTensor(x, mask), NestedTensor(x_pos, mask)]
            
        return out
        
        
class PointNetAbstractionBackbone(PointNetAbstractionBackboneBase):
    
    def __init__(self, pre_trained_backbone: str, reduction : list, reduction_map : dict, point_feature_dim : int = 6, use_conv : bool = True, use_propagation : bool = True, pointnet_dropout_rate : float = 0.2):
        
        backbone = PointNetAbstraction(point_feature_dim = point_feature_dim, use_conv = use_conv, use_propagation = use_propagation, dropout_rate = pointnet_dropout_rate, checkpoint_path = pre_trained_backbone)
        super().__init__(backbone, reduction, reduction_map)        

        
# ...
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        
        pos = []
        for name, x in xs.items():
            
            # Position Encoding:
            if isinstance(x, list):
                [x, x_pos] = x
                
                out.append(x)
                ret = self[1](x_pos, fixed_positions = False)
                
            else:
                
                out.append(x)
                ret = self[1](x)
            
            # ...
            if isinstance(ret, tuple):
                p_emb, m_emb = ret
                pos.append([p_emb.to(x.tensors.dtype), m_emb.to(x.tensors.dtype)])
            
            else:
                pos.append(ret.to(x.tensors.dtype))
                
        return out, pos

# ...
def build_backbone(config):
    
    # ...
    backbone_type = config.get("backbone")
    
    # ...
    if "efficientnet" in backbone_type:
        
        position_embedding = build_position_encoding(config)
        backbone = EfficientNetBackbone(backbone_type, config.get("reduction"))

    # ...
    elif "pointnet" in backbone_type:
        
        # ...
        point_feature_dim = config.get("point_feature_dim") if "point_feature_dim" in config.keys() else 6
        
        use_conv = config.get("use_conv") if "use_conv" in config.keys() else True
        use_propagation = config.get("use_propagation") if "use_propagation" in config.keys() else True
        
        pointnet_dropout_rate = config.get("pointnet_dropout_rate") if "pointnet_dropout_rate" in config.keys() else 0.2
    
        # ...
        dimension_aligned_position_embedding = config.get("dimension_aligned_position_embedding") if "dimension_aligned_position_embedding" in config.keys() else False
        aligned_embedding_dimensions = config.get("aligned_embedding_dimensions") if "aligned_embedding_dimensions" in config.keys() else None
    
        # ...
        position_embedding = build_position_encoding(config, nr_coord = 3, dimension_aligned_position_embedding = dimension_aligned_position_embedding, aligned_embedding_dimensions = aligned_embedding_dimensions)
        backbone = PointNetAbstractionBackbone(config.get("backbone_ckpt"), config.get("reduction"), config.get("reduction_map"), 
                                               point_feature_dim = point_feature_dim, use_conv = use_conv, use_propagation = use_propagation, pointnet_dropout_rate = pointnet_dropout_rate)
        
    else:
        raise NotImplementedError(f"Backbone Type {backbone_type} isn't supported yet.")
        
    # ...
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    
    return model
