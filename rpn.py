import numpy as np
from torch.nn import functional as F
import torch
from torch import nn

from torchinfo import summary
from creator import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_sizes=[128, 256, 512], feat_stride=16, proposal_creator_params=dict(),):
        super(RegionProposalNetwork, self).__init__()

        # zero-centered anchor
        #
        #  (-ws,-hs)------
        #    |           |
        #    |   (0,0)   |
        #    |           |
        #    ----------(ws,hs)
        #
        #                 0       1        2       3
        # image format: batch, channels, height, width

        ratios       = torch.as_tensor(ratios,       dtype=torch.float32, device=torch.device("cpu"))
        anchor_sizes = torch.as_tensor(anchor_sizes, dtype=torch.float32, device=torch.device("cpu"))
        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * anchor_sizes[None, :]).view(-1)
        hs = (h_ratios[:, None] * anchor_sizes[None, :]).view(-1)

        self.base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        self.base_anchors = self.base_anchors.round()


        self.feat_stride = feat_stride
        self.n_anchor = self.base_anchors.shape[0]
        self.conv1   = nn.Conv2d(in_channels,  mid_channels, 3, 1, 1)
        self.relu    = nn.ReLU()
        self.score   = nn.Conv2d(mid_channels, self.n_anchor * 2, 1, 1, 0)
        self.loc     = nn.Conv2d(mid_channels, self.n_anchor * 4, 1, 1, 0)


    def forward(self, x):
        n, _, hh, ww = x.shape

        x = self.conv1(x)
        x = self.relu(x)

        # location regression
        rpn_locs = self.loc(x)               

        # fg/bg classification
        rpn_scores = self.score(x)
        return rpn_locs, rpn_scores



def test():
    rpn_inst = RegionProposalNetwork()
    summary(rpn_inst)
    
    x = torch.zeros([8, 512, 50, 40], dtype=torch.float32)
    rpn_inst.eval()
    loc_output, cls_output = rpn_inst(x, x.shape[2:])
    print(loc_output.shape)
    print(cls_output.shape)


if __name__=='__main__':
    test()
