import torch.nn.functional as F
from torch import nn
from .pencoder import build_position_encoding, NestedTensor
from typing import Dict, List
import torch
import timm

class BackboneBase(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', num_classes=0)
        self.reductions = reduction
        self.reduction_map = {"layer3": 1024, "layer4": 2048}
        self.num_channels = [self.reduction_map[reduction] for reduction in self.reductions]
    def extract_feats(self, x):
        for params in self.backbone.parameters():
            params.requires_grad = False
        with torch.no_grad():
            feats = self.backbone.conv1(x)
            feats = self.backbone.relu(self.backbone.bn1(feats))
            feats = self.backbone.maxpool(feats)
            feats_layer1 = self.backbone.layer1(feats)
            feats_layer2 = self.backbone.layer2(feats_layer1)
            feats_layer3 = self.backbone.layer3(feats_layer2)
            feats_layer4 = self.backbone.layer4(feats_layer3)
            return {'layer3': feats_layer3, 'layer4':feats_layer4}
        
    def forward(self, tensor_list: NestedTensor):
        xs = self.extract_feats(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name in self.reductions:
            x = xs[name]
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            ret = self[1](x)
            if isinstance(ret, tuple):
                p_emb, m_emb = ret
                pos.append([p_emb.to(x.tensors.dtype), m_emb.to(x.tensors.dtype)])
            else:
                pos.append(ret.to(x.tensors.dtype))

        return out, pos

def build_backbone(config):
    position_embedding = build_position_encoding(config)
    backbone = BackboneBase(config.get("reduction"))
    model = Joiner(backbone, position_embedding)
    return model
