import os, torch
from PIL import Image
import math, cv2, random
import numpy as np
import albumentations as Aug
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataset.utils.obj_transformations import draw_umich_gaussian, gaussian_radius

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)

NAME_TO_ID = {
    'Pedestrian':0,
    'person':0,
    'Car':1,
    'Van':1,
    'car':1,
    'trailer':1,
    'Cyclist':2,
    'bicycle':2,
    'motorcycle':2,
    'rider':2,
    'Truck': 3,
    'bus':3,
    'truck':3,
    'Tram':4,
    'train':4,
    'Person_sitting':-1,
    'Misc':-1,
    'DontCare':-1
}



class ObjectDataset(Dataset):
    def __init__(self, config, mode):
        super(ObjectDataset, self).__init__()

        self.config = config
        self.num_classes = self.config['arch']['num_obj_classes']
        self.max_objs = self.config['arch']['max_obj_number']

        self.mode = mode
        self.specific_dataset = config['datasets']['obj'] + '_dataloader'
        self.dataset_dir = config[self.specific_dataset]['args']['data_dir']

        self.input_size = [self.config['arch']['input_width'], self.config['arch']['input_height']]
        self.input_w = self.input_size[0]
        self.input_h = self.input_size[1]
        self.output_w = self.input_w // 4
        self.output_h = self.input_h // 4

        if self.mode == 'test':
            self.img_path = os.path.join(self.data_dir, 'testing', 'image')
        else:
            self.annot_path = os.path.join(self.dataset_dir, 'training', 'label')
            self.img_path = os.path.join(self.dataset_dir, 'training', 'image')
            self.split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', 'train.txt' if self.mode == 'train' else 'val.txt')
            self.sample_id_list = [x.strip() for x in open(self.split_txt_path).readlines()]
            if self.config['datasets']['obj'] == 'bdd100k' and self.mode=='val':
                random.shuffle(self.sample_id_list)
                self.sample_id_list = self.sample_id_list[:1500]
        self.gaussian_iou = 0.7

        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.num_samples = len(self.sample_id_list)

        self.offset = 0 if config['datasets']['obj'] == 'kitti' else 3

        print('Loaded %d %s Kitti samples' % (self.num_samples, self.mode))


    def __getitem__(self, index):

        # Get paths
        img_id = self.sample_id_list[index]
        if self.config['datasets']['obj'] == 'kitti':
            img_path = os.path.join(self.img_path, '{:06d}.png'.format(int(img_id)))
            annotations = os.path.join(self.annot_path, '{:06d}.txt'.format(int(img_id)))
        else:
            img_path = os.path.join(self.img_path, img_id + '.jpg')
            annotations = os.path.join(self.annot_path, img_id + '.txt')


        # Get image and labels
        img = Image.open(img_path)
        labels, boxes = self.get_labels(annotations)

        width, height = img.size
        if len(boxes) == 0:
            boxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])

        if self.mode == 'train':

            transform = Aug.Compose([Aug.HorizontalFlip(p=self.config['datasets']['augmentations']['flip']),
                                     Aug.MotionBlur(blur_limit=10, p=self.config['datasets']['augmentations']['motionBlur']),
                                     Aug.ShiftScaleRotate(scale_limit=[-0.2, 0.3] if self.config['datasets']['augmentations']['scale'] else [0.0,0.0],
                                                          shift_limit = 0.3 if self.config['datasets']['augmentations']['shift'] else 0.0,
                                                          rotate_limit= 7.0 if self.config['datasets']['augmentations']['rotate'] else [0.,0.],
                                                          border_mode=0,
                                                          interpolation=cv2.INTER_LINEAR,
                                                          p=self.config['datasets']['augmentations']['shiftScaleRotate'])],
                                    bbox_params = Aug.BboxParams(format='pascal_voc', label_fields=['category_ids']),)

            transformed = transform(image=np.array(img), bboxes=boxes, category_ids=labels)
            img, boxes, labels = Image.fromarray(transformed['image']), transformed['bboxes'], np.array(transformed['category_ids'])

        img = self.img_transform(img)

        hmap = np.zeros((self.num_classes, self.output_h, self.output_w), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        for k, (bbox, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = [elem for elem in bbox]

            x1, x2 = x1*self.input_w/float(width)/4., x2*self.input_w/float(width)/4.
            y1, y2 = y1*self.input_h/float(height)/4.,  y2*self.input_h/float(height)/4.

            h, w = y2 - y1, x2 - x1

            if h > 0 and w > 0:
                obj_c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                hmap[label] = draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.output_w + obj_c_int[0]
                ind_masks[k] = 1


        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks}#'img_id': img_id}

    def __len__(self):
        return self.num_samples

    def get_labels(self,annotation):
        """
        annotation: path to annotation files
        returns bounding box coordinates in image frame
        """
        labels = []
        boxes = []
        for line in open(annotation, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]
            cat_id = NAME_TO_ID[obj_name]
            if cat_id < 0:  # ignore Tram and Misc
                continue
            bbox = np.array(
                [float(line_parts[4-self.offset]),
                 float(line_parts[5-self.offset]),
                 float(line_parts[6-self.offset]),
                 float(line_parts[7-self.offset])])

            labels.append(cat_id)
            boxes.append(bbox)

        return np.asarray(labels), np.asarray(boxes)

