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



def label_assignment(anchor, targets, img, scale):
    height, width, _ = img.shape
    gt_bbox   = len(targets)
    num_anchor = anchor.shape[0]
    tbl = np.zeros((gt_bbox,num_anchor))
    logging.info(f"# of gt bboxes: {gt_bbox}   # of anchors: {num_anchor}")

    fg_cls_label = np.full(num_anchor,-1)
    reg_label = np.zeros((num_anchor,4))

    start = time.time()
    for i in range(0,gt_bbox):
        obj = targets[i]
        #bbox = [float(b.numpy()) for b in obj['bbox']]
        #print("debug: ", obj['bbox'])
        bbox = obj['bbox']
        x,y,w,h = [a*scale for a in bbox]

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

    fg_cls_label = gen_mini_batch(fg_cls_label,256)
    logging.info(f"# of fg anchors: {np.count_nonzero(fg_cls_label == 1)}")
    logging.info(f"# of bg anchors: {np.count_nonzero(fg_cls_label == 0)}")
    logging.info(f"# of dont care anchors: {np.count_nonzero(fg_cls_label == -1)}")


    fg_cls_label = torch.from_numpy(fg_cls_label).to(device)
    reg_label = torch.from_numpy(reg_label).to(device)

    return raw_fg_cls_label, fg_cls_label, reg_label


def rescale(imgs,targets, side_len):
    bsize, _, hh, ww = imgs.shape

    #print(f"  hh: {hh}    ww:{ww}")
    scale = side_len/ww if hh > ww else side_len/hh
    if hh > ww:
        new_hh = int(hh * scale + 0.5)
        new_ww = side_len
    else:
        new_hh = side_len
        new_ww = int(ww * scale + 0.5)

    #imgs = transforms.Resize(imgs, side_len)
    #print(f" scale: {scale} new hh {new_hh}    new ww: {new_ww}")
    imgs = torch.nn.functional.interpolate(imgs,size=(new_hh,new_ww), mode='bilinear')


    return imgs, scale


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

    num = 0

    rpn_cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    #rpn_loc_criterion = nn.SmoothL1Loss()
    #rpn_loc_criterion = nn.L1Loss()
    rpn_loc_criterion = nn.L1Loss(reduction='none')


    # lr=0.002 no convergence ~ 30K overfitting?
    # lr=0.01 no convergence for fg/bg overfitting?
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
    train_loss = 0
    cls_loss = 0
    reg_loss = 0
    scale = 1.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        #inputs, scale = rescale(inputs,targets,640)


        optimizer.zero_grad()

        logging.debug(f"Running on {device}")
        inputs = inputs.to(device)
        batch_size = inputs.shape[0]
        #logging.info(f"intput: {inputs}")
        #logging.info(f"intput shape: {inputs.shape}")
        loc_output, cls_output, anchor = net(inputs)


        logging.debug("output shape: ")
        logging.debug(f"    loc: {loc_output.shape}")
        logging.debug(f"    cls: {cls_output.shape}")
        #logging.info(f"Scale: {scale} and all anchors: {anchor.shape}")

        img = inputs.permute(0,2,3,1).cpu().numpy()[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width, _ = img.shape

        #cv2.imshow(' ', img)
        #cv2.waitKey()


        cls_output = cls_output.permute(0, 2, 3, 1).contiguous().view(-1,2)
        loc_output = loc_output.permute(0, 2, 3, 1).contiguous().view(-1,4)


        #logging.info(f"Total label assignment runtime: {end-start}")
        #check_bbox(targets, img, num)

        start = time.time()
        raw_fg_cls_label, fg_cls_label, reg_label = label_assignment(anchor, targets, img, scale)
        end = time.time()
        logging.debug(f"cls_output: {cls_output.shape}     fg_cls_label: {fg_cls_label.shape}")
        logging.info(f"Total label assignment runtime: {end-start}")
        logging.info(f"    Learning Rate: {optimizer.param_groups[0]['lr']}")
    


        #logging.info(f"    debug fg_cls_label: {fg_cls_label.shape}")
        fg_cls_loss = rpn_cls_criterion(cls_output, fg_cls_label)

        mask = np.array([[1,1,1,1] if raw_fg_cls_label[idx] == 1 else [0,0,0,0] for idx in range(0,anchor.shape[0])])
        mask = torch.from_numpy(mask).to(device)

        #mask = np.zeros(anchor.shape[0])
        #for idx in range(0,anchor.shape[0]):
        #    if raw_fg_cls_label[idx] == 1:
        #        print("debug: ", raw_fg_cls_label[idx])
        #        mask[idx]=1

        num_pos = np.count_nonzero(raw_fg_cls_label == 1)
        logging.info(f"Number of randomly picked positive anchors for loc regression: {num_pos}")


        if num_pos > 0: 
            selected_idx = np.where(raw_fg_cls_label==1)[0]

            loc_output.register_hook(lambda grad: grad * mask.float())
            rpn_loc_loss = rpn_loc_criterion(loc_output.float(),reg_label.float())
            rpn_loc_loss = torch.mean(rpn_loc_loss[selected_idx])


            #print(np.where(raw_fg_cls_label==1))
            #logging.info(f"    Regression labels for pos one: {reg_label[selected_idx]}")
            #rpn_loc_loss = rpn_loc_criterion(loc_output[selected_idx], reg_label[selected_idx])
            print("rpn_loc_loss: ", rpn_loc_loss)

        
            _, predicted = cls_output.max(1)

            total = 0
            correct = 0
            for i, label in enumerate(predicted):
                if fg_cls_label[i]!=-1:
                    total += 1
                if fg_cls_label[i]==label:
                    correct += 1

            logging.debug(f"    debug: {selected_idx}")
            logging.debug(f"    debug: {selected_idx[0]}")
            logging.debug(f"    debug: {selected_idx.shape}")
            logging.info(f"    Sample cross entropy of pos one: {cls_output[selected_idx[0]]}")
            logging.info(f"    Total: {total} correct: {correct}   Accu. : {correct/total}")
 

            #logging.info(f"Sample loc regression {loc_output[raw_fg_cls_label.tolist().index(1)]}   mask: {mask[raw_fg_cls_label.tolist().index(1)]}")
            total_loss = fg_cls_loss+rpn_loc_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            cls_loss += fg_cls_loss.item()
            reg_loss += rpn_loc_loss.item()

            rpn_loc_loss_val = rpn_loc_loss.item()
        else:
            total_loss = fg_cls_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            cls_loss += fg_cls_loss.item()

            rpn_loc_loss_val = 0

        avg = train_loss/(batch_idx+1)
        avg_cls = cls_loss/(batch_idx+1)
        avg_reg = reg_loss/(batch_idx+1)
        logging.info(f"    {batch_idx}. Ave. train loss: {avg}    average cls loss: {avg_cls}               average reg loss: {avg_reg}")
        logging.info(f"                                                current cls loss: {fg_cls_loss.item()}    current reg loss: {rpn_loc_loss_val}")
        logging.info("---------------------------------------------------")

        #cv2.imshow(' ', img)
        #cv2.waitKey()

        #if num==2:
        #    break

        if num>0 and num%10000==0:
            torch.save( net.state_dict(), os.path.join( "./savedModels/",'fasterRCNN_itr_'+str(num)+'.pth') )

        num += 1

    #summary(resnet_50)
    #model = torchvision.models.resnet50()
    #summary(model)

    print("-----------------------------")



if __name__ == '__main__':
    loadLabels()
    train()
