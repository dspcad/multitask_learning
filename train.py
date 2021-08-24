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



def train():
    print("    1. Construct Faster-RCNN")
    resnet_50 = ResNet_large(ResidualBlockBottleneck, [3, 4, 6, 3]).to(device)
    rpn_inst = RegionProposalNetwork(2048, feat_stride=32)

    net = FasterRCNN(resnet_50, rpn_inst)
    net = net.to(device)

    print("    2. Load coco train2017")
    transform_train = transforms.Compose([transforms.ToTensor()])
    coco_train = CocoDetection("/home/hhwu/datasets/coco/train2017", "/home/hhwu/datasets/coco/annotations/instances_train2017.json", transform=transform_train)
    dataloader = DataLoader(coco_train, batch_size=1, shuffle=True, num_workers=0)

    num = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(device)
        inputs = inputs.to(device)
        print("intput shape: ", inputs.shape)
        loc_output, cls_output = net(inputs)
        print("output shape: ")
        print("    loc: ", loc_output.shape)
        print("    cls: ", cls_output.shape)
        anchor = rpn_inst._generated_all_anchor(inputs.shape[2],inputs.shape[3])
        print("All anchors: ", anchor.shape)

        img = inputs.permute(0,2,3,1).cpu().numpy()[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, _ = img.shape


        gt_bbox   = len(targets)
        num_anchor = anchor.shape[0]
        tbl = np.zeros((gt_bbox,num_anchor))
        print(f"debug: # of gt bboxes {gt_bbox}")
        fg_cls_label = np.full(num_anchor,-1)
        reg_label = np.zeros((num_anchor,4))
        for i in range(0,gt_bbox):
            obj = targets[i]
            #bbox = [float(b.numpy()) for b in obj['bbox']]
            #print("debug: ", obj['bbox'])
            bbox = obj['bbox']
            x,y,w,h = [int(a+0.5) for a in bbox]
            x1,y1,x2,y2 = x,y,x+w,y+h

            for j in range(0,num_anchor):
                tbl[i][j] = IoU([x1,y1,x2,y2], anchor[j])

                wa = anchor[j][2]-anchor[j][0]
                ha = anchor[j][3]-anchor[j][1]
                xa = anchor[j][0]+wa/2
                ya = anchor[j][1]+ha/2
                #tx
                reg_label[j][0] = (x-xa)/wa
                #ty
                reg_label[j][1] = (y-ya)/wa
                #tw
                reg_label[j][2] = np.log(w/wa)
                #th
                reg_label[j][3] = np.log(h/ha)



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

                    print(f"    reg label:  {reg_label[j]}")


            idx = np.argmax(tbl[i])
            if tbl[i][idx] != 0:
                fg_cls_label[idx] = 1
                anchor_x1 = int(max(anchor[idx][0],0))
                anchor_y1 = int(max(anchor[idx][1],0))
                anchor_x2 = int(min(anchor[idx][2],width-1))
                anchor_y2 = int(min(anchor[idx][3],height-1))
                cate_id = int(obj['category_id'].numpy())
                cv2.rectangle(img, (anchor_x1, anchor_y1), (anchor_x2, anchor_y2), (255,0,0), 1)
                print("    anchor > 0.0: {}     {} {} {} {}".format(categories[cate_id], anchor_x1, anchor_y1, anchor_x2, anchor_y2))
                cv2.putText(img, "{} {} {} {}".format(anchor_x1, anchor_y1, anchor_x2, anchor_y2), (anchor_x1, anchor_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                print(f"    reg label:  {reg_label[idx]}")


        for j in range(0,num_anchor):
            idx = np.argmax(tbl[:,j])
            if tbl[idx][j] < 0.3:
                #print(f"{j} anchor:      iou:{tbl[idx][j]}")
                if fg_cls_label[j] != 1:
                    fg_cls_label[j] = 0
                #print(f"{j} anchor:      label:{fg_cls_label[idx]}")

        print("# of fg anchors: ", np.count_nonzero(fg_cls_label == 1))
        print("# of bg anchors: ", np.count_nonzero(fg_cls_label == 0))
        print("# of dont care anchors: ", np.count_nonzero(fg_cls_label == -1))



        cv2.imshow(' ', img)
        cv2.waitKey()

        if num==10:
            break

        num += 1

    #summary(resnet_50)
    #model = torchvision.models.resnet50()
    #summary(model)

    print("-----------------------------")



if __name__ == '__main__':
    loadLabels()
    train()
