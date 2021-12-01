import torch.utils.data as data
from PIL import Image
import os
import os.path

class resize(object):
    def __init__(self, h=640, w=640):
        from pycocotools.coco import COCO
        self.target_h = h
        self.target_w = w
    def __call__(self, img, target):
        #print(f"target h: {self.target_h}")
        #print(f"target w: {self.target_w}")
        #print(f"img size: {img.size}")

        # PIL image has the size in (width, height)
        ww, hh = img.size

        gt_bbox  = len(target)
        if gt_bbox != 0:
            for i in range(0,gt_bbox):
                scale_x = self.target_w/ww
                scale_y = self.target_h/hh
  
                if int(target[i]['bbox'][2]+0.5)!=0 and int(target[i]['bbox'][3]+0.5)!=0:
                    target[i]['bbox'][0] = target[i]['bbox'][0]*scale_x
                    target[i]['bbox'][1] = target[i]['bbox'][1]*scale_y
                    target[i]['bbox'][2] = target[i]['bbox'][2]*scale_x
                    target[i]['bbox'][3] = target[i]['bbox'][3]*scale_y


        return target




class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.target_transform is not None:
            target = self.target_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        print(f"coco transformed img size: {img.shape}")
        return img, target


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
