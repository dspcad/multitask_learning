import torch
import torchvision
import torch.nn as nn


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn

        self.roi_pooling = nn.AdaptiveMaxPool2d((7,7))
        # First fully connected layer 7x7x2048 -> 1024
        self.fc = nn.Linear(100352, 1024)

        # head for the classification
        self.fc_cls = nn.Linear(1024, 81)
        # head for the location 81x4
        self.fc_loc = nn.Linear(1024, 324)


        nn.init.normal_(self.fc.weight, 0, 0.01)
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
        valid_rois = valid_rois[nms_res]
        batch_rois = []
        for roi in valid_rois:
            #x ,y is the center of roi
            x, y, w, h = roi
            x, y, w, h = int(x+0.5), int(y+0.5), int(w+0.5), int(h+0.5)

            #print(f"debug: x:{x}   y:{y}  w:{w}  h:{h}   ww: {ww}  hh: {hh}")
            x1 = (x-w/2)/32
            y1 = (y-h/2)/32
            x2 = (x+w/2)/32
            y2 = (y+h/2)/32

            x1 = int(min(max(x1,0),ww-1))
            y1 = int(min(max(y1,0),hh-1))
            x2 = int(min(max(x2,1),ww))
            y2 = int(min(max(y2,1),hh))

            x1 = min(x1,x2-1)
            y1 = min(y1,y2-1)
            #print(f"debug: x1:{x1}   y1:{y1}  x2:{x2}  y2:{y2}   roi shape: {roi.shape}")

            roi = feat_map[0,:,y1:y2,x1:x2]


            roi = self.roi_pooling(roi)
            batch_rois.append(roi)

        batch_rois = torch.stack(batch_rois)
        #print(f"debug: batch_rois: {torch.tensor(batch_rois).shape}")
        batch_rois = batch_rois.contiguous().view(len(nms_res),-1)
        #batch_rois = batch_rois.contiguous().view(len(valid_rois),-1)
        print(f"debug: batch_rois: {batch_rois.shape}")
        out        = self.fc(torch.tensor(batch_rois))
        roi_scores = self.fc_cls(out)
        roi_locs   = self.fc_loc(out)
         
        return rpn_locs, rpn_scores, self.rpn.anchor, roi_locs, roi_scores, nms_res




