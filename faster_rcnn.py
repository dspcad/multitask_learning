import torch
import torch.nn as nn




class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn


    def forward(self, x):
        h = self.extractor(x)
        rpn_locs, rpn_scores = self.rpn(h)

        return rpn_locs, rpn_scores, self._generated_all_anchor(x.shape[2], x.shape[3])



    def _generated_all_anchor(self, height, width):
        cell_x = torch.arange(0, width,  self.rpn.feat_stride)
        cell_y = torch.arange(0, height, self.rpn.feat_stride)
        cell_x, cell_y = torch.meshgrid(cell_x, cell_y)
        cell = torch.stack((cell_x.ravel(), cell_y.ravel(), cell_x.ravel(), cell_y.ravel()), axis=1)

        A = self.rpn.base_anchors.shape[0]
        K = cell.shape[0]
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        # return (K*A, 4)
        anchor = self.rpn.base_anchors.reshape((1, A, 4)) + cell.reshape((1, K, 4)).permute((1, 0, 2))
        anchor = anchor.reshape((K * A, 4))
        return anchor

