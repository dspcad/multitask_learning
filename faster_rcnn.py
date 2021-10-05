import torch
import torchvision
import torch.nn as nn


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn

        self.roi_pooling = nn.AdaptiveMaxPool2d((7,7))
        # First fully connected layer 7x7x2048 -> 4096
        self.fc1 = nn.Linear(100352, 4096)
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


    def forward(self, x, cls_label):
        feat_map = self.extractor(x)
        _, ch, hh, ww = feat_map.shape

        rpn_locs, rpn_scores, rois = self.rpn(feat_map)
        valid_rois = rois[cls_label != -1]
        #fg_cls_score = fg_cls_score[:,1]
        print(f"debug faster rcnn: {valid_rois.shape}")
        #print(f"debug: {rpn_scores[index_inside,1].shape}")

        nms_res = torchvision.ops.nms(valid_rois, rpn_scores[cls_label != -1,1],0.6)
        batch_rois = []
        valid_rois = valid_rois[nms_res]
        for roi in valid_rois:
            x, y, w, h = roi
            x, y, w, h = int(x+0.5), int(y+0.5), int(w+1), int(h+1)

            if x<0 or x>=ww or y<0 or y>=hh or w<1 or x+w>ww or h<1 or y+h>hh:
                #roi = feat_map[0,:,0:1,0:1]
                roi = feat_map[0]
            else:
                roi = feat_map[0,:,y:y+h,x:x+w]

            roi = self.roi_pooling(roi)
            #print(f"debug: x:{x}   y:{y}  w:{w}  h:{h}   roi shape: {roi.shape}")
            batch_rois.append(roi)

        batch_rois = torch.stack(batch_rois)
        #print(f"debug: batch_rois: {torch.tensor(batch_rois).shape}")
        batch_rois = batch_rois.contiguous().view(len(nms_res),-1)
        print(f"debug: batch_rois: {batch_rois.shape}")
        out        = self.fc1(torch.tensor(batch_rois))
        out        = self.fc2(out)
        roi_scores = self.fc_cls(out)
        roi_locs   = self.fc_loc(out)
         
        return rpn_locs, rpn_scores, self.rpn.anchor, rois, roi_locs, roi_scores, nms_res




