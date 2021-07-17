import torch
import torch.nn as nn
import torchvision

from torchinfo import summary
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from resnet import *
from rpn import *
from faster_rcnn import FasterRCNN
import cv2, math
import numpy as np
from skimage.transform import rescale

import sys
np.set_printoptions(threshold=sys.maxsize)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadLabels():
    print("---------- Load coco.name ---------")
    global categories
    with open("coco.names", "r") as f:
    
        categories = f.read().split("\n")
        categories = [x for x in categories if x]
        categories.insert(0,'')
        print(categories)
    print("-----------------------------")



def IoU(bb1=list(),bb2=list()):
    assert len(bb1) == 4
    assert len(bb2) == 4

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou


class Rescale(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size 
        self.max_size = max_size

    def __call__(self, sample):
        print("debug: ",sample)
        img, targets = sample
        _, H, W = img.shape

        img = self.preprocess(img)
        _, o_H, o_W = img.shape
        scale = o_H / H

#        for i in range(0,len(targets)):
#            obj = targets[i]
#            bbox = [float(b.numpy()) for b in obj['bbox']]
#            x,y,w,h = [int(a) for a in bbox]
#            x1,y1,x2,y2 = x,y,x+w,y+h


        bbox = self.resize_bbox(targets, (H, W), (o_H, o_W))


        return img, targets

    def preprocess(self, img):
        C, H, W = img.shape
        scale1 = self.min_size / min(H, W)
        scale2 = self.max_size / max(H, W)
        scale  = min(scale1, scale2)
        img    = rescale(img, scale, anti_aliasing=False)

        return img

    def resize_bbox(self, targets, in_size, out_size):
        bbox = bbox.copy()
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        bbox[:, 0] = y_scale * bbox[:, 0]
        bbox[:, 2] = y_scale * bbox[:, 2]
        bbox[:, 1] = x_scale * bbox[:, 1]
        bbox[:, 3] = x_scale * bbox[:, 3]

        return bbox
        

def rescaleImg(img, min_size=600, max_size=1000):
    print("      ----- resize -----")
    print("      Before: ",img.shape)
    H, W, C = img.shape
    scale  = min_size / min(H, W)
    W = min(W*scale, 1000)
    H = min(H*scale, 1000)

    #img     = rescale(img, scale, anti_aliasing=False)

    img     = cv2.resize(img, (int(W+0.5), int(H+0.5)), interpolation = cv2.INTER_AREA)
    print("      After:  ",img.shape)
    print("      ------------------")

    return img

def rescaleBBox(bbox, in_size, out_size):
    w_scale = float(out_size[0]) / in_size[0]
    h_scale = float(out_size[1]) / in_size[1]
    bbox[0] = int(w_scale * bbox[0]+0.5) 
    bbox[1] = int(h_scale * bbox[1]+0.5)
    bbox[2] = int(w_scale * bbox[2]+0.5)
    bbox[3] = int(h_scale * bbox[3]+0.5)
 
    return bbox


def rpn_loss():
    print("---------- RPN LOSS ---------")

    print("    1. load coco train2017")
    transform_train = transforms.Compose([transforms.ToTensor()])
    coco_train = CocoDetection("/home/hhwu/Datasets/COCO/train2017", "/home/hhwu/Datasets/COCO/annotations_trainval2017/annotations/instances_train2017.json", transform=transform_train)
    dataloader = DataLoader(coco_train, batch_size=1, shuffle=True, num_workers=0)
    # print(type(coco_train))

    print("    2. generate base anchors")
    ratios=[0.5, 1, 2]
    anchor_sizes=[128, 256, 512]
    h_ratios = np.sqrt(ratios)
    w_ratios = 1 / h_ratios
    print("      ratios       : ", ratios)
    print("      anchor sizes=: ", anchor_sizes)


    ws = np.outer(w_ratios, anchor_sizes).flatten()
    hs = np.outer(h_ratios, anchor_sizes).flatten()


    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    print("      = base anchors =")
    for a in base_anchors:
        print("         ", a)




    print("    3. display some samples of COCO")
    A = base_anchors.shape[0]
    num = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        #inputs, targets = inputs.to(device), targets.to(device)
        print("      pytorch tensor: batch, channels, height, width")
        print("      batch: {batch:}   image shape: {input_shape:}".format(batch=batch_idx, input_shape=inputs.shape))
        img = inputs.permute(0,2,3,1).numpy()[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        raw_h, raw_w, _ = img.shape

        img = rescaleImg(img)
        height, width, _ = img.shape


        feat_h, feat_w = math.ceil(height/16), math.ceil(width/16)
        print("      feat stride: 16")
        print("      feat shape:  {} {}".format(feat_h, feat_w))

  
        
        cell_x = np.arange(0, width,  16)
        cell_y = np.arange(0, height, 16)
        cell_x, cell_y = np.meshgrid(cell_x, cell_y)
        cell = np.stack((cell_x.ravel(), cell_y.ravel(), cell_x.ravel(), cell_y.ravel()), axis=1)
        K = cell.shape[0]
        print("cell: ", K)

        anchor = base_anchors.reshape((1, A, 4)) + cell.reshape((1, K, 4)).transpose((1, 0, 2))
        anchor = anchor.reshape((K * A, 4))
        print(anchor.shape)

        num_anchor = K*A
        num_bbox   = len(targets)
        tbl = np.zeros((num_bbox,num_anchor))
        fg_cls_label = np.full(num_anchor,-1)

        for i in range(0,num_bbox):
            obj = targets[i]
            bbox = [float(b.numpy()) for b in obj['bbox']]
            x,y,w,h = [int(a) for a in bbox]
            x1,y1,x2,y2 = x,y,x+w,y+h
            x1,y1,x2,y2 = rescaleBBox([x1,y1,x2,y2], (raw_w, raw_h), (width,height))

            for j in range(0,num_anchor):
                tbl[i][j] = IoU([x1,y1,x2,y2], anchor[j])

                if(tbl[i][j]>0.7):
                    #print(" bbox: {}    anchor:{}   {:.2f}".format(i,j,tbl[i][j]))
                    fg_cls_label[j] = 1

                    anchor_x1 = int(max(anchor[j][0],0))
                    anchor_y1 = int(max(anchor[j][1],0))
                    anchor_x2 = int(min(anchor[j][2],width-1))
                    anchor_y2 = int(min(anchor[j][3],height-1))
                    cate_id = int(obj['category_id'].numpy())
                    cv2.rectangle(img, (anchor_x1, anchor_y1), (anchor_x2, anchor_y2), (255,0,0), 1)
                    print("    anchor > 0.7: {}     {} {} {} {}".format(categories[cate_id], anchor_x1, anchor_y1, anchor_x2, anchor_y2))
                    cv2.putText(img, "{} {} {} {}".format(anchor_x1, anchor_y1, anchor_x2, anchor_y2), (anchor_x1, anchor_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            idx = np.argmax(tbl[i])
            if tbl[i][idx] != 0:
                fg_cls_label[idx] = 1
                anchor_x1 = int(max(anchor[idx][0],0))
                anchor_y1 = int(max(anchor[idx][1],0))
                anchor_x2 = int(min(anchor[idx][2],width-1))
                anchor_y2 = int(min(anchor[idx][3],height-1))
                cate_id = int(obj['category_id'].numpy())
                cv2.rectangle(img, (anchor_x1, anchor_y1), (anchor_x2, anchor_y2), (255,0,0), 1)
                print("    anchor > 0.7: {}     {} {} {} {}".format(categories[cate_id], anchor_x1, anchor_y1, anchor_x2, anchor_y2))
                cv2.putText(img, "{} {} {} {}".format(anchor_x1, anchor_y1, anchor_x2, anchor_y2), (anchor_x1, anchor_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


            #print("# of fg anchors: ", np.count_nonzero(fg_cls_label == 1))

        max_iou = np.amax(tbl,axis=0)
        for j in range(0,num_anchor):
            if fg_cls_label[j] == -1 and max_iou[j]<0.3:
                fg_cls_label[j] = 0

        print("# of fg anchors: {} ".format(np.count_nonzero(fg_cls_label == 1)))
        print("# of bg anchors: ", np.count_nonzero(fg_cls_label == 0))
#        for obj in targets:
#            cate_id = int(obj['category_id'].numpy())
#            bbox = [float(b.numpy()) for b in obj['bbox']]
#            x,y,w,h = [int(a) for a in bbox]
#            print("{}: {} {}".format(cate_id, categories[cate_id], [x,y,w,h]))
#            x1,y1,x2,y2 = x,y,x+w,y+h
#            x1,y1,x2,y2 = rescaleBBox([x1,y1,x2,y2], (raw_w, raw_h), (width,height))
#            print("scaled: {}: {} {}".format(cate_id, categories[cate_id], [x1,y1,x2,y2]))
#            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
#            #cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
#            cv2.putText(img, "{}".format(categories[cate_id]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow(' ', img)
        cv2.waitKey()


        if num==1:
            break

        num = num + 1


    resnet_50 = ResNet_large(ResidualBlockBottleneck, [3, 4, 6, 3]).to(device)
    rpn_inst = RegionProposalNetwork()

    net = FasterRCNN(resnet_50, rpn_inst)
    net = net.to(device)
    #summary(resnet_50)
    #model = torchvision.models.resnet50()
    #summary(model)

    print("-----------------------------")



if __name__ == '__main__':
    loadLabels()
    rpn_loss()
