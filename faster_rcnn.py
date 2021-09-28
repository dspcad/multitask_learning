import torch
import torch.nn as nn




class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, roi):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.roi = roi


    def forward(self, x):
        h = self.extractor(x)

        self.rpn._generated_all_anchor(x.shape[2], x.shape[3])
        rpn_locs, rpn_scores, roi = self.rpn(h)
  
         
        return rpn_locs, rpn_scores, self.rpn.anchor, roi




