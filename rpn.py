import numpy as np
from torch.nn import functional as F
import torch
from torch import nn

from torchinfo import summary



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
        ratios       = torch.as_tensor(ratios,       dtype=torch.float32, device=torch.device("cpu"))
        anchor_sizes = torch.as_tensor(anchor_sizes, dtype=torch.float32, device=torch.device("cpu"))
        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * anchor_sizes[None, :]).view(-1)
        hs = (h_ratios[:, None] * anchor_sizes[None, :]).view(-1)

        self.base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        self.base_anchors = self.base_anchors.round()


        self.feat_stride = feat_stride
        #self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        self.n_anchor = self.base_anchors.shape[0]
        self.conv1   = nn.Conv2d(in_channels,  mid_channels, 3, 1, 1)
        self.relu    = nn.ReLU()
        self.score   = nn.Conv2d(mid_channels, self.n_anchor * 2, 1, 1, 0)
        self.loc     = nn.Conv2d(mid_channels, self.n_anchor * 4, 1, 1, 0)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _generated_all_anchors(hh*self.feat_stride, ww*self.feat_stride)

        x = self.conv1(x)
        x = self.relu(x)

        # location regression
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # fg/bg classification
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, self.n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        #for i in range(n):
        #    roi = self.proposal_layer(
        #        rpn_locs[i].cpu().data.numpy(),
        #        rpn_fg_scores[i].cpu().data.numpy(),
        #        anchor, img_size,
        #        scale=scale)
        #    batch_index = i * np.ones((len(roi),), dtype=np.int32)
        #    rois.append(roi)
        #    roi_indices.append(batch_index)

        #rois = np.concatenate(rois, axis=0)
        #roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    def _generated_all_anchor(height, width):
        cell_x = torch.arange(self.feat_stride/2, width,  self.feat_stride)
        cell_y = torch.arange(self.feat_stride/2, height, self.feat_stride)
        cell_x, cell_y = torch.meshgrid(cell_x, cell_y)
        cell = torch.stack((cell_x.ravel(), cell_y.ravel(), cell_x.ravel(), cell_y.ravel()), axis=1)

        A = self.base_anchors[0]
        K = cell.shape[0]
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # return (K*A, 4)
        anchor = anchor_base.reshape((1, A, 4)) + cell.reshape((1, K, 4)).permute((1, 0, 2))
        anchor = anchor.reshape((K * A, 4))
        return anchor

rpn_inst = RegionProposalNetwork()
summary(rpn_inst)
