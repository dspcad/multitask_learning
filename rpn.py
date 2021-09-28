import numpy as np
from torch.nn import functional as F
import torch
from torch import nn

from torchinfo import summary
from creator import ProposalCreator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_sizes=[64, 128, 256, 512], feat_stride=16):
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

        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.score.weight, 0, 0.01)
        nn.init.normal_(self.loc.weight, 0, 0.01)


    def forward(self, x):
        n, _, hh, ww = x.shape

        x = self.conv1(x)
        x = self.relu(x)

        # location regression
        rpn_locs = self.loc(x)               

        # fg/bg classification
        rpn_scores = self.score(x)


        cls_output = rpn_scores.permute(0, 2, 3, 1).contiguous().view(-1,2)
        loc_output = rpn_locs.permute(0, 2, 3, 1).contiguous().view(-1,4)


        #print(f"debug: rpn anchor:   {self.anchor.shape}")
        # xa, ya, wa, ha
        center_anchor = [[(a[2]+a[0])/2, (a[3]+a[1])/2, (a[2]-a[0])/2, (a[3]-a[1])/2] for a in self.anchor]
        print(f"debug: rpn center anchor:   {len(center_anchor)}     ron_locs: {rpn_locs.shape}")
        roi = torch.Tensor([[loc[0]*center_anchor[idx][2]+center_anchor[idx][0], loc[1]*center_anchor[idx][3]+center_anchor[idx][1], torch.exp(loc[2])*center_anchor[idx][2], torch.exp(loc[3])*center_anchor[idx][3]]  for idx, loc in enumerate(loc_output)]).to(device)

        return loc_output, cls_output, roi

    def _generated_all_anchor(self, height, width):
        cell_x = torch.arange(0, width,  self.feat_stride)
        cell_y = torch.arange(0, height, self.feat_stride)
        cell_x, cell_y = torch.meshgrid(cell_x, cell_y)
        cell = torch.stack((cell_x.ravel(), cell_y.ravel(), cell_x.ravel(), cell_y.ravel()), axis=1)

        A = self.base_anchors.shape[0]
        K = cell.shape[0]
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # return (K*A, 4)
        self.anchor = self.base_anchors.reshape((1, A, 4)) + cell.reshape((1, K, 4)).permute((1, 0, 2))
        self.anchor = self.anchor.reshape((K * A, 4))
        return self.anchor



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
