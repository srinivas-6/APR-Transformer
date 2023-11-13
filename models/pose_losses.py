import torch
import torch.nn.functional as F
import torch.nn as nn


class CameraPoseLoss(nn.Module):
    """
    A class to represent camera pose loss
    """

    def __init__(self, config):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(CameraPoseLoss, self).__init__()
        self.learnable = config.get("learnable")
        self.s_x = torch.nn.Parameter(torch.Tensor([config.get("s_x")]), requires_grad=self.learnable)
        self.s_q = torch.nn.Parameter(torch.Tensor([config.get("s_q")]), requires_grad=self.learnable)
        self.norm = config.get("norm")

    def forward(self, est_pose, gt_pose):
            """
            Forward pass
            :param est_pose: (torch.Tensor) batch of estimated poses, a Nx7 tensor
            :param gt_pose: (torch.Tensor) batch of ground_truth poses, a Nx7 tensor
            :return: camera pose loss
            """
            # Position loss
            l_x = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=self.norm).mean()
            # Orientation loss (normalized to unit norm)
            l_q = torch.norm(F.normalize(gt_pose[:, 3:], p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                             dim=1, p=self.norm).mean()

            if self.learnable:
                return l_x * torch.exp(-self.s_x) + self.s_x + l_q * torch.exp(-self.s_q) + self.s_q
            else:
                return self.s_x*l_x + self.s_q*l_q



# Cross Entropy Loss adapted from meetshah1995 to prevent size inconsistencies between model precition 
# and target label
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction='mean', ignore_index=255
    )
    return loss