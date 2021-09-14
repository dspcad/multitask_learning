import torch
import torch.nn as nn
import torchvision

from torchinfo import summary
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from resnet import *
from rpn import *
from faster_rcnn import FasterRCNN
import cv2, math
import numpy as np

import logging, sys, time, os
logging.basicConfig(format='%(levelname)-4s %(message)s',level=logging.INFO)

np.set_printoptions(threshold=sys.maxsize)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadLabels():
    print("---------- Load coco.name ---------")
    global categories
    with open("coco.names", "r") as f:

        categories = f.read().split("\n")
        categories = [x for x in categories if x]
        categories.insert(0,'')
        logging.info("COCO Dataset classes: {}".format(len(categories)))
    print("-----------------------------")


def IoU(bb1=list(),bb2=list()):
    assert len(bb1) == 4
    assert len(bb2) == 4


    if bb1[0] >= bb1[2] or bb1[1] >= bb1[3] or bb2[0] >= bb2[2] or bb2[1] >= bb2[3]:
        print(f"BBox1: {bb1}  BBox2 : {bb2}")
    #print(f"BBox1: {bb1}  BBox2 : {bb2}")

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


#def gen_mini_batch(fg_cls_label,batch_size=128):
#    res_idx_0 = np.array(np.where(fg_cls_label == 0)).flatten()
#    res_idx_1 = np.array(np.where(fg_cls_label == 1)).flatten()
#
#    logging.info("Generate mini batch: ")
#    logging.info("    Before: ")
#    logging.info(f"    # of pos: {len(res_idx_1)}")
#    logging.info(f"    # of neg: {len(res_idx_0)}")
#    if len(res_idx_1)>batch_size/2:
#        non_selected_idx_of_pos = np.random.choice(res_idx_1,len(res_idx_1)-int(batch_size/2),replace=False)
#        for idx in non_selected_idx_of_pos:
#            fg_cls_label[idx]=-1
#
#    non_selected_idx_of_neg = np.random.choice(res_idx_0, len(res_idx_0)-batch_size + min(int(batch_size/2),len(res_idx_1)),replace=False)
#    for idx in non_selected_idx_of_neg:
#        fg_cls_label[idx]=-1
#
#    res_idx_0 = np.array(np.where(fg_cls_label == 0)).flatten()
#    res_idx_1 = np.array(np.where(fg_cls_label == 1)).flatten()
#
#    logging.info("    After: ")
#    logging.info(f"    # of pos: {len(res_idx_1)}")
#    logging.info(f"    # of neg: {len(res_idx_0)}")
#
#    return fg_cls_label


def gen_mini_batch(fg_cls_label,batch_size=128):
    res_idx_0 = np.array(np.where(fg_cls_label == 0)).flatten()
    res_idx_1 = np.array(np.where(fg_cls_label == 1)).flatten()

    logging.info("Generate mini batch: ")
    logging.info("    Before: ")
    logging.info(f"    # of pos: {len(res_idx_1)}")
    logging.info(f"    # of neg: {len(res_idx_0)}")


    batch_size = min(len(res_idx_0),len(res_idx_1))

    if len(res_idx_0) > batch_size:
        non_selected_idx_of_neg = np.random.choice(res_idx_0, len(res_idx_0)-batch_size,replace=False)
        for idx in non_selected_idx_of_neg:
            fg_cls_label[idx]=-1


    if len(res_idx_1) > batch_size:
        non_selected_idx_of_pos = np.random.choice(res_idx_1, len(res_idx_1)-batch_size,replace=False)
        for idx in non_selected_idx_of_pos:
            fg_cls_label[idx]=-1


    res_idx_0 = np.array(np.where(fg_cls_label == 0)).flatten()
    res_idx_1 = np.array(np.where(fg_cls_label == 1)).flatten()

    logging.info("    After: ")
    logging.info(f"    # of pos: {len(res_idx_1)}")
    logging.info(f"    # of neg: {len(res_idx_0)}")

    return fg_cls_label



def label_assignment(anchor, target, img, scale_x, scale_y):
    # img is pytorch tensor
    # batch, channel, height, width
    _, height, width = img.shape
    gt_bbox   = len(target)
    num_anchor = anchor.shape[0]
    tbl = np.zeros((gt_bbox,num_anchor))
    logging.info(f"# of gt bboxes: {gt_bbox}   # of anchors: {num_anchor}")

    fg_cls_label = np.full(num_anchor,-1)
    reg_label = np.zeros((num_anchor,4))

    start = time.time()
    for i in range(0,gt_bbox):
        obj = target[i]
        #bbox = [float(b.numpy()) for b in obj['bbox']]
        #print("debug: ", obj['bbox'])
        bbox = obj['bbox']
        #x,y,w,h = [a*scale for a in bbox]
        x,y,w,h = bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y

        x1,y1,x2,y2 = int(x+0.5),int(y+0.5),int(x+w+0.5),int(y+h+0.5)
        if x1==x2 or y1==y2:
            logging.info(f"WARNNING:    x1:{x1} y1:{y1} x2:{x2} y2:{y2}")
            continue

        #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        #cate_id = int(obj['category_id'].numpy())
        #logging.info("    {}     {} {} {} {}".format(categories[cate_id], x1, y1, x2, y2))
        #cv2.putText(img, "{}".format(categories[cate_id]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        for j in range(0,num_anchor):
            wa = anchor[j][2]-anchor[j][0]
            ha = anchor[j][3]-anchor[j][1]
            xa = anchor[j][0]+wa/2
            ya = anchor[j][1]+ha/2

            #tbl[i][j] = IoU([x1,y1,x2,y2], anchor[j]) if abs(xa-x)*2< (w+wa) and abs(ya-y)*2 < (h+ha) else 0
            if abs(xa-x)*2< (w+wa) and abs(ya-y)*2 < (h+ha):
                tbl[i][j] = IoU([x1,y1,x2,y2], anchor[j])

                #tx
                reg_label[j][0] = (x-xa)/wa
                #ty
                reg_label[j][1] = (y-ya)/wa
                #tw
                reg_label[j][2] = np.log(w/wa)
                #th
                reg_label[j][3] = np.log(h/ha)





            

        #foreground: the highest IoU with a gt box
        #foreground: IoU > 0.7 with any gt box
        max_v = np.max(tbl[i])
        if max_v > 0:
            fg_cls_label[np.logical_or(tbl[i]>0.5, tbl[i] == max_v)] = 1

        #for j in range(0,num_anchor):
        #    if tbl[i][j] == max_v or tbl[i][j]>0.7:
        #        fg_cls_label[j] = 1
        #        anchor_x1 = int(max(anchor[j][0],0))
        #        anchor_y1 = int(max(anchor[j][1],0))
        #        anchor_x2 = int(min(anchor[j][2],width-1))
        #        anchor_y2 = int(min(anchor[j][3],height-1))
        #        cate_id = int(obj['category_id'].numpy())
        #        cv2.rectangle(img, (anchor_x1, anchor_y1), (anchor_x2, anchor_y2), (255,0,0), 1)
        #        logging.debug("    anchor > 0.0: {}     {} {} {} {}".format(categories[cate_id], anchor_x1, anchor_y1, anchor_x2, anchor_y2))
        #        cv2.putText(img, "{} {} {} {}".format(anchor_x1, anchor_y1, anchor_x2, anchor_y2), (anchor_x1, anchor_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        #        logging.debug(f"    reg label:  {reg_label[j]}")


    end = time.time()
    logging.info(f"    FG selection runtime {end-start}")


    start = time.time()
    if gt_bbox == 0:
        logging.debug(f"    No gt bbox: {gt_bbox}")
        fg_cls_label = np.full(num_anchor,0)

    else:
        #background: IoU < 0.3 for all gt boxes
        for j in range(0,num_anchor):
            #idx = np.argmax(tbl[:,j])
            max_v = np.max(tbl[:,j])
            #if tbl[idx][j] < 0.1 and fg_cls_label[j] != 1:
            #if tbl[idx][j] == 0:
            if max_v == 0:
                fg_cls_label[j] = 0

    end = time.time()
    logging.info(f"    BG selection runtime {end-start}")

    raw_fg_cls_label = np.copy(fg_cls_label)
    #print(f"   raw pos: {np.count_nonzero(raw_fg_cls_label)}")
    #print(f"       pos: {np.count_nonzero(fg_cls_label)}")

    fg_cls_label = gen_mini_batch(fg_cls_label,256)
    logging.info(f"# of fg anchors: {np.count_nonzero(fg_cls_label == 1)}")
    logging.info(f"# of bg anchors: {np.count_nonzero(fg_cls_label == 0)}")
    logging.info(f"# of dont care anchors: {np.count_nonzero(fg_cls_label == -1)}")


    fg_cls_label = torch.from_numpy(fg_cls_label).to(device)
    reg_label = torch.from_numpy(reg_label).to(device)

    return raw_fg_cls_label, fg_cls_label, reg_label


#def rescale(imgs,targets, side_len):
#    bsize, _, hh, ww = imgs.shape
#
#    #print(f"  hh: {hh}    ww:{ww}")
#    scale = side_len/ww if hh > ww else side_len/hh
#    if hh > ww:
#        new_hh = int(hh * scale + 0.5)
#        new_ww = side_len
#    else:
#        new_hh = side_len
#        new_ww = int(ww * scale + 0.5)
#
#
#    #imgs = transforms.Resize(imgs, side_len)
#    #print(f" scale: {scale} new hh {new_hh}    new ww: {new_ww}")
#    imgs = torch.nn.functional.interpolate(imgs,size=(new_hh,new_ww), mode='bilinear')
#
#
#    return imgs, scale

def rescale(img, side_len):
    bsize, _, hh, ww = img.shape

    scale_x = side_len/ww
    scale_y = side_len/hh

    new_ww = int(ww * scale_x + 0.5)
    new_hh = int(hh * scale_y + 0.5)

    img = torch.nn.functional.interpolate(img,size=(new_hh,new_ww), mode='bilinear')


    return img[0], scale_x, scale_y

def check_bbox(targets, img, num):
    for obj in targets:
        #bbox = [float(b.numpy()) for b in obj['bbox']]
        #print("debug: ", obj['bbox'])
        bbox = obj['bbox']
        x,y,w,h = [a for a in bbox]
        cate_id = int(obj['category_id'].numpy())
        x1,y1,x2,y2 = int(x+0.5),int(y+0.5),int(x+w+0.5),int(y+h+0.5)

        if w==0 or h==0:
            #logging.info(f"{obj['id']}    {categories[cate_id]}: {x} {y} {w} {h}");
            logging.info(f"{obj['image_id']}    {categories[cate_id]}: {x} {y} {w} {h}");
            
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            #cv2.imshow(' ', img)
            #cv2.waitKey()

        #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        #cate_id = int(obj['category_id'].numpy())
        #logging.info("    {}     {} {} {} {}".format(categories[cate_id], x1, y1, x2, y2))


def trainOneEpoch(dataloader, net, optimizer, rpn_cls_criterion, rpn_loc_criterion, epoch):
    train_loss = 0
    cls_loss = 0
    reg_loss = 0
    batch_size = 4
    batch_imgs = []
    scale_x = []
    scale_y = []
    targets = []

    for batch_idx, (img, target) in enumerate(dataloader):
        if batch_idx > 0 and batch_idx % batch_size==0:

            ################################################
            #   feed mini batch of images to Faster-RCNN   #
            ################################################
            optimizer.zero_grad()
            batch_imgs = torch.stack(batch_imgs).to(device)
            logging.debug(f"debug batch_imgs: {batch_imgs.shape}")
            loc_output, cls_output, anchor = net(batch_imgs)

            cls_output = cls_output.permute(0, 2, 3, 1).contiguous().view(-1,2)
            loc_output = loc_output.permute(0, 2, 3, 1).contiguous().view(-1,4)
            logging.debug(f"debug cls_output: {cls_output.shape}")
            logging.debug(f"debug loc_output: {loc_output.shape}")

            #########################################################
            #   Generate the labels for RPN cls loss and loc loss   #
            #########################################################
            raw_fg_cls_label = []
            fg_cls_label     = []
            reg_label        = []
            for i in range(0,batch_size):
                raw_fg_cls_label_1_img, fg_cls_label_1_img, reg_label_1_img = label_assignment(anchor, targets[i], batch_imgs[i], scale_x[i], scale_y[i])
                raw_fg_cls_label.append(raw_fg_cls_label_1_img)
                fg_cls_label.append(fg_cls_label_1_img)
                reg_label.append(reg_label_1_img)


            #mask = []
            #for i, raw_fg_cls_label_1_img in enumerate(raw_fg_cls_label):
            #    mask_1_img = [[1,1,1,1] if raw_fg_cls_label_1_img[idx] == 1 else [0,0,0,0] for idx in range(0,anchor.shape[0])]
            #    mask.append(mask_1_img)


            #########################################################
            #    loc loss:  Mask the gradient computations of neg   #
            #########################################################
            #mask = torch.Tensor(mask).to(device)
            #mask = np.array(mask)
            #mask = torch.from_numpy(mask).view(-1,4).to(device)
            #loc_output.register_hook(lambda grad: grad * mask.float())
            reg_label = torch.stack(reg_label).contiguous().view(-1,4).to(device)
            raw_fg_cls_label = torch.from_numpy(np.array(raw_fg_cls_label)).flatten().to(device)

            rpn_loc_loss = rpn_loc_criterion(loc_output.float(), reg_label.float())
            rpn_loc_loss_1         = rpn_loc_loss[raw_fg_cls_label == 1]
            rpn_loc_loss_0         = rpn_loc_loss[raw_fg_cls_label == 0].zero_()
            rpn_loc_loss_dont_care = rpn_loc_loss[raw_fg_cls_label == -1].zero_()
            #rpn_loc_loss_1.retain_grad()
            #rpn_loc_loss_0.retain_grad()
            #rpn_loc_loss_dont_care.retain_grad()

            #logging.info(f"debug raw_fg_cls_label: {torch.count_nonzero(raw_fg_cls_label)}")
            #print("debug: rpn_loc_loss", rpn_loc_loss.grad_fn)
            #print("debug: rpn_loc_loss_1", rpn_loc_loss_1.grad_fn)
            #print("debug: rpn_loc_loss_0", rpn_loc_loss_0.grad_fn)

            logging.debug("loss_1: ", torch.mean(rpn_loc_loss_1).item())
            logging.debug("loss_0: ", torch.mean(rpn_loc_loss_0).item())
            logging.debug("loss_dont_care: ", torch.mean(rpn_loc_loss_dont_care).item())
            #rpn_loc_loss = torch.mean(rpn_loc_loss_1) + torch.mean(rpn_loc_loss_0) + torch.mean(rpn_loc_loss_dont_care)
            #rpn_loc_loss = torch.mean(rpn_loc_loss_1)
            rpn_loc_loss = rpn_loc_loss_1.mean()
            #print("debug: rpn_loc_loss", rpn_loc_loss.grad_fn)
            #rpn_loc_loss.retain_grad()
            #rpn_loc_loss = rpn_loc_criterion(loc_output[raw_fg_cls_label==1].float(),reg_label[raw_fg_cls_label==1].float())

            #################
            #    cls loss   #
            #################
            #fg_cls_label = torch.stack(fg_cls_label).contiguous().view(-1,1).to(device)
            fg_cls_label = torch.flatten(torch.stack(fg_cls_label)).to(device)
            fg_cls_loss  = rpn_cls_criterion(cls_output, fg_cls_label)
            #print(f"debug fg_cls_loss: {fg_cls_loss.grad_fn}")



            total_loss = (fg_cls_loss + 2*rpn_loc_loss)
            #print("debug: totoalloss", total_loss.grad_fn)

            total_loss.backward()
            #print(f"debug rpn_loc_loss_1: {rpn_loc_loss_1.grad}")
            #print(f"debug rpn_loc_loss_0: {rpn_loc_loss_0.grad}")
            #print(f"debug rpn_loc_loss_dont_care: {rpn_loc_loss_dont_care.grad}")
            #print(f"debug rpn_loc_loss: {rpn_loc_loss.grad}")
            optimizer.step()
           

            train_loss += total_loss.item()
            cls_loss += fg_cls_loss.item()
            reg_loss += rpn_loc_loss.item()

            total = 0

            # max value, index
            _, predicted = cls_output.max(1)
            correct = 0
            for i, label in enumerate(predicted):
                if fg_cls_label[i]!=-1:
                    total += 1
                if fg_cls_label[i]==label:
                    correct += 1

            num_batch = batch_idx/batch_size
            avg = train_loss/num_batch
            avg_cls = cls_loss/num_batch
            avg_reg = reg_loss/num_batch
            logging.info(f"------------ Batch Training Result (Epoch {epoch})----------------")
            logging.info(f"    {batch_idx}. Ave. train loss: {avg}    average cls loss: {avg_cls}               average reg loss: {avg_reg}")
            logging.info(f"                                              current cls loss: {fg_cls_loss.item()}               current reg loss: {rpn_loc_loss.item()}")
            logging.info(f"    Total: {total} correct: {correct}   Accu. : {correct/total}")
            logging.info("---------------------------------------------------")

            #####################
            #    reset batch    #
            #####################
            batch_imgs = []
            scale_x = []
            scale_y = []
            targets = []





        img, scale_xx, scale_yy = rescale(img, 800)
        batch_imgs.append(img)
        scale_x.append(scale_xx)
        scale_y.append(scale_yy)
        targets.append(target)


        #cv2.imshow(' ', img)
        #cv2.waitKey()

        #if num==2:
        #    break

        if batch_idx >0 and batch_idx%10000==0:
            torch.save( net.state_dict(), os.path.join( "./savedModels/",'fasterRCNN_itr_'+str(batch_idx)+'.pth') )




def train():
    logging.info("    1. Construct Faster-RCNN")
    resnet_50 = ResNet_large(ResidualBlockBottleneck, [3, 4, 6, 3]).to(device)
    rpn_inst = RegionProposalNetwork(2048, feat_stride=32)

    net = FasterRCNN(resnet_50, rpn_inst)
    net = net.to(device)

    logging.info("    2. Load coco train2017")
    #transform_train = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    coco_train = CocoDetection("/home/hhwu/datasets/coco/train2017", "/home/hhwu/datasets/coco/annotations/instances_train2017.json", transform=transform_train, target_transform=None)
    dataloader = DataLoader(coco_train, batch_size=1, shuffle=True, num_workers=0)


    rpn_cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    #rpn_loc_criterion = nn.SmoothL1Loss()
    #rpn_loc_criterion = nn.L1Loss()
    rpn_loc_criterion = nn.L1Loss(reduction='none')


    # lr=0.002 no convergence ~ 30K overfitting?
    # lr=0.01 no convergence for fg/bg overfitting?
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    #summary(resnet_50)
    #model = torchvision.models.resnet50()
    #summary(model)

    for epoch in range(1,5):
        trainOneEpoch(dataloader, net, optimizer, rpn_cls_criterion, rpn_loc_criterion, epoch)



if __name__ == '__main__':
    loadLabels()
    train()
