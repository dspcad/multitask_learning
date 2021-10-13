import torch
import torch.nn as nn
import torchvision

from torchinfo import summary
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from resnet import *
from rpn import *
from roi import *
from train import *
from faster_rcnn import FasterRCNN
import cv2, math
import numpy as np

import logging, sys, time, os
logging.basicConfig(format='%(levelname)-4s %(message)s',level=logging.INFO)

np.set_printoptions(threshold=sys.maxsize)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def validate():
    transform_valid = transforms.Compose([transforms.ToTensor()])
    coco_valid = CocoDetection("/home/hhwu/datasets/coco/val2017", "/home/hhwu/datasets/coco/annotations/instances_val2017.json", transform=transform_valid)
    dataloader = DataLoader(coco_valid, batch_size=1, shuffle=True, num_workers=0)

    resnet_50 = ResNet_large(ResidualBlockBottleneck, [3, 4, 6, 3])
    rpn_inst = RegionProposalNetwork(2048, feat_stride=32)

    net = FasterRCNN(resnet_50, rpn_inst)
    net = net.to(device)
    logging.info(f"------------ Loading Faster RCNN ----------")
    net.load_state_dict(torch.load("./savedModels/fasterRCNN_itr_50000.pth"))
    net.eval()
    
    num=0
    for img, target in dataloader:
        #img, scale_x, scale_y = rescale(img, 600)
        print(f"img: {img.shape}")
        res = cv2.cvtColor(img[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        cv2.imshow(' ', res)
        cv2.waitKey()

        img = img.to(device)
        net.rpn._generated_all_anchor(img.shape[2], img.shape[3])
        loc_output, cls_output, anchor, roi_locs, roi_scores, nms_res = net(img)

        num +=1

        if num==5:
            break


if __name__ == '__main__':
    validate()
