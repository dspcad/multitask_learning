import torch
import torch.nn as nn




class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn


    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores = self.rpn(h, img_size, scale)
        return rpn_locs, rpn_scores
