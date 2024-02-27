"""
Code for the position encoding of TransPoseNet
 code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- changed to learn also the position of a learned pose token
"""
import torch
from torch import nn
from typing import Optional
from torch import Tensor

import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import numpy as np

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
# if float(torchvision.__version__[:3]) < 0.7:
#     from torchvision.ops import _new_empty_tensor
#     from torchvision.ops.misc import _output_size

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    
    # TODO: Make this more general:
    if tensor_list[0].ndim == 3:
        
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO: Make it support different-sized images:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        
        # ...
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        
        # ...
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
       
    # ...
    elif tensor_list[0].ndim == 2:
        
        if torchvision._is_tracing():
            raise NotImplementedError(f'ONXX nested tensor are not support for the 2-dim case (currently no 3-dim support available).')
    
        max_size = _max_by_axis([list(vectors.shape) for vectors in tensor_list])
        
        # ...
        batch_shape = [len(tensor_list)] + max_size
        b, c, w = batch_shape
        
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        
        # ...
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, w), dtype=torch.bool, device=device)
        
        for vector, pad_vector, m in zip(tensor_list, tensor, mask):
            pad_vector[: vector.shape[0], : vector.shape[1]].copy_(vector)
            m[: vector.shape[1]] = False
    
    # ...
    else:
        raise ValueError('not supported')
        
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

# ...
# TODO: Implement changes to PositionEmbeddingLearned also in PositionEmbeddingLearnedWithPoseToken:
class PositionEmbeddingLearnedWithPoseToken(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(60, num_pos_feats)
        self.col_embed = nn.Embedding(60, num_pos_feats)
        self.pose_token_embed = nn.Embedding(60, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.pose_token_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) + 1
        j = torch.arange(h, device=x.device) + 1
        p = i[0]-1
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        p_emb = torch.cat([self.pose_token_embed(p),self.pose_token_embed(p)]).repeat(x.shape[0], 1)

        # embed of position in the activation map
        m_emb = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return p_emb, m_emb

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256, ndim=2, dimension_aligned_position_embedding=False, aligned_embedding_dimensions=None, device = None):
        super().__init__()
        
        # ...
        self.device = device
        
        # ...
        self.embeddings = []
        for dim in range(ndim):
            self.embeddings.append(nn.Embedding(50, num_pos_feats) if self.device is None else nn.Embedding(50, num_pos_feats, device = self.device))
        
        self.reset_parameters()
        
        # ...
        self.dimension_alignment = dimension_aligned_position_embedding
        self.aligned_embedding_dimensions = aligned_embedding_dimensions

    def reset_parameters(self):
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight)

    def forward(self, tensor_list: NestedTensor, fixed_positions: bool = True):
        
        # ...
        x = tensor_list.tensors
        
        # ...
        x_dim = x.shape # = [Batch_Size, Channels, ...]
        x_feature_shape = x_dim[2:] 
        
        # ...
        if fixed_positions:
        
            # ...
            assert len(x_feature_shape) == len(self.embeddings)
        
            # ...
            x_positions = []
            for feature_dim in x_feature_shape:
                x_positions.append(torch.arange(feature_dim, device=x.device))
                
            # ...
            pos = [embed(x_positions[idx]).unsqueeze(idx).repeat(
                                                *[x_feature_shape[idx] if idx2 == idx else 1 for idx2 in range(len(x_dim)-1)]) for idx, embed in enumerate(self.embeddings)]
            
            pos = torch.cat(pos, dim=-1)
            pos = pos.permute(*[len(x_dim)-2 if idx3 == 0 else idx3-1 for idx3 in range(len(x_dim)-1)])
            pos = pos.unsqueeze(0)
            pos = pos.repeat(*[x.shape[0] if idx4 == 0 else 1 for idx4 in range(len(x_dim))]) 

        # ...
        else:
            
            # ...
            if self.dimension_alignment:    
            # TODO: Currently only #channels == #dims, i.e. points as xyz arrays. In further update make it possible that #dim's == #shape_dims_after_channels (e.g. img -> 2 dim's);
        
                # ... 
                [nr_batches, nr_dimensions, nr_points] = x.shape

                if self.aligned_embedding_dimensions is None:
                    dimension_axis_length : int = round(pow(nr_points, 1/nr_dimensions)) # TODO: Implement Sanity Checks, if dim size can be used to divide current vector;
                
                else:
                    dimension_axis_length : list = self.aligned_embedding_dimensions
                    assert np.prod(dimension_axis_length) == nr_points, "Product of specified dimensions in 'aligend embedding dimensions' parameter must be equal the number of points (3rd shape value of x)."
                    assert len(dimension_axis_length) == nr_dimensions, "Number of specified dimensions in 'aligend embedding dimensions' parameter must be equal the number of shape dimensions (2nd shape value of x)."

                # ...
                # TODO: Fix case of same values in first dimension -> wrong ordering when considering the second dimension;
                if nr_dimensions == 2:

                    # ...
                    if type(dimension_axis_length) is int:
                        [dimension_axis_length_x, dimension_axis_length_y] = [dimension_axis_length] * nr_dimensions 

                    else:
                        [dimension_axis_length_x, dimension_axis_length_y] = dimension_axis_length 

                    # ...
                    out = torch.full(x.shape, -1)
                    
                    # ...
                    _, xi = x[:, 0].sort()[1].sort()
                    xii = torch.floor(torch.div(xi, dimension_axis_length_x))
                    
                    # ...
                    out[:, 0] = xii
                    for rank in range(dimension_axis_length_y):
                        out[:, 1][xii == rank] = x[:, 1][xii == rank].view((nr_batches, -1)).sort()[1].sort()[1].view((-1))

                    out = out
                        
                    # ...
                    pos = [embed(out[:, idx]) for idx, embed in enumerate(self.embeddings)] 
                    pos = torch.cat(pos, dim=-1)
                    pos = pos.permute(*[0] + [len(x_dim)-1 if idx3 == 0 else idx3 for idx3 in range(len(x_dim[1:]))])
                        
                elif nr_dimensions == 3:
                    
                    # ...
                    if type(dimension_axis_length) is int:
                        [dimension_axis_length_x, dimension_axis_length_y, dimension_axis_length_z] = [dimension_axis_length] * nr_dimensions 

                    else:
                        [dimension_axis_length_x, dimension_axis_length_y, dimension_axis_length_z] = dimension_axis_length 

                    # ...
                    out = torch.full(x.shape, -1)
                    x_ = x.to(out.device)

                    # ...
                    _, xi = x_[:, 0].sort()[1].sort()
                    xii = torch.floor(torch.div(xi, dimension_axis_length_y*dimension_axis_length_z))
                    
                    # ...         
                    out[:, 0] = xii
                    
                    for rank in range(dimension_axis_length_x):
                        
                        out_x = x_[:, 1][xii == rank]
                        out_x = out_x.view((nr_batches, -1))
                        out_x = out_x.sort()[1].sort()[1]
                        out_x = out_x.view((-1))
                        out_x = torch.floor(torch.div(out_x, dimension_axis_length_z))
                            
                        out[:, 1][xii == rank] = out_x.long()
                                                                                    
                    # ...
                    for rank_x in range(dimension_axis_length_x):
                        for rank_y in range(dimension_axis_length_y):
                            
                            out[:, 2][(xii == rank_x).logical_and(out[:, 1] == rank_y)] = x_[:, 2][(xii == rank_x).logical_and(out[:, 1] == rank_y)].view((nr_batches, -1)).sort()[1].sort()[1].view((-1))
                    
                    out = out.to(x.device)
                        
                    # ...
                    pos = [embed(out[:, idx]) for idx, embed in enumerate(self.embeddings)] 
                    pos = torch.cat(pos, dim=-1)
                    pos = pos.permute(*[0] + [len(x_dim)-1 if idx3 == 0 else idx3 for idx3 in range(len(x_dim[1:]))])

                else:
                    raise NotImplementedError(f'Sorted Position Embeddings are only supported for the 2-dim and 3-dim case (currently no generic support available).')

            # ...
            else:
                
                # ...
                # TODO: TEST: Sort works differently (gives list of [index of element with lowest value, second lowest value, ...]). We want [ranking of first element, ...]
                x_sorted, x_sorted_indicies = x.sort() 
                x_sorted_indicies = x_sorted_indicies.sort()[1]
                
                # ...
                pos = [embed(x_sorted_indicies[:, idx]) for idx, embed in enumerate(self.embeddings)] # 16, 16, 128
                pos = torch.cat(pos, dim=-1)
                pos = pos.permute(*[0] + [len(x_dim)-1 if idx3 == 0 else idx3 for idx3 in range(len(x_dim[1:]))])

        return pos # = [Batch_Size, Embedding_Size, ... with ... -> e.g. ... nr_points] or ... width, height] or etc.

    
def build_position_encoding(config, nr_coord : int = 2, dimension_aligned_position_embedding : bool = False, aligned_embedding_dimensions : list = None):
    
    # ...
    hidden_dim = config.get("hidden_dim")
    N_steps = hidden_dim // nr_coord
    
    device = config.get("device_id") if "device_id" in config.keys() else None
    device = "cpu" if "cuda" in device and not torch.cuda.is_available() else device

    # ...
    learn_embedding_with_pose_token = config.get("learn_embedding_with_pose_token")
    if learn_embedding_with_pose_token:
        
        if nr_coord != 2:
            raise NotImplementedError(f'Position Embeddings with Pose Tokens are only supported for the 2-dim case (currently no 3-dim support available).')
        
        position_embedding = PositionEmbeddingLearnedWithPoseToken(N_steps)
        
    else:
        position_embedding = PositionEmbeddingLearned(N_steps, ndim = nr_coord, dimension_aligned_position_embedding = dimension_aligned_position_embedding, aligned_embedding_dimensions = aligned_embedding_dimensions, device = device)

    return position_embedding
