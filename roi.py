import numpy as np
from torch.nn import functional as F
import torch
from torch import nn

from torchinfo import summary


class ROIHeadlNetwork(nn.Module):
    def __init__(self):
        super(ROIHeadlNetwork, self).__init__()

        self.roi_pooling = nn.AdaptiveMaxPool2d((7,7))
        # First fully connected layer 7x7x512 -> 4096
        self.fc1 = nn.Linear(25088, 4096)
        # Second fully connected layer 4096x4096
        self.fc2 = nn.Linear(4096, 4096)

        # head for the classification
        self.fc_cls = nn.Linear(4096, 81)
        # head for the location 81x4
        self.fc_loc = nn.Linear(4096, 324)


        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.normal_(self.fc_loc.weight, 0, 0.01)


    def forward(self, proposal):
        n, _, hh, ww = proposal.shape
        print(f"debug proposal: {proposal.shape}")

        out = self.roi_pooling(proposal)
        print(f"debug roi out: {out.shape}")
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        # location regression
        roi_locs   = self.fc_loc(out)               
        # classification
        roi_scores = self.fc_cls(out)               

        return roi_locs, roi_scores



def test():
    roi_inst = ROIHeadlNetwork()
    summary(roi_inst)
    
    x = torch.zeros([8, 512, 50, 40], dtype=torch.float32)
    roi_inst.eval()
    loc_output, cls_output = roi_inst(x)
    print(f"loc: {loc_output.shape}")
    print(f"cls: {cls_output.shape}")


if __name__=='__main__':
    test()
