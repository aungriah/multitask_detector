import os, torch, pdb, wandb
from PIL import Image
import math, cv2, random
import numpy as np
import albumentations as Aug
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataset.utils.obj_transformations import draw_umich_gaussian, gaussian_radius

from dataset.utils.augmentations import find_start_pos, FreeScaleMask, MaskToTensor
from dataset.utils.lane_anchors import tusimple_row_anchor

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)

NAME_TO_ID = {
    'car':1,
    'truck':3,
    'pedestrian':0,
    'cyclist':2,
    'rider':2,
    'tram':4,
    'motorcycle':2
}

class JointDataset(Dataset):
    def __init__(self, config, mode):
        super(JointDataset, self).__init__()

        self.config = config
        self.num_classes = self.config['arch']['num_obj_classes']
        self.max_objs = self.config['arch']['max_obj_number']

        self.mode = mode
        self.dataset_dir = config['finetuning_dataloader']['args']['data_dir']

        # Constants for object detection
        self.input_size = [self.config['arch']['input_width'], self.config['arch']['input_height']]
        self.input_width = self.input_size[0]
        self.input_height = self.input_size[1]
        self.obj_output_width = self.input_width // 4
        self.obj_output_height = self.input_height // 4
        self.lane_output_width = self.input_width // 8
        self.lane_output_height = self.input_height // 8
        self.gaussian_iou = 0.7

        # Constants for lane detection
        self.griding_num = config['finetuning_dataloader']['args']['griding_num']
        self.num_lanes = config['finetuning_dataloader']['args']['num_lanes']
        self.use_aux = config['arch']['use_aux']
        self.row_anchor = tusimple_row_anchor
        self.row_anchor.sort()

        # Image path
        self.img_path = os.path.join(self.dataset_dir, 'training', 'image')
        self.img_label_path = os.path.join(self.dataset_dir, 'training', 'segLabel')

        # Annotation path
        self.annot_path = os.path.join(self.dataset_dir, 'training', 'label')


        # Split for objects
        self.split_txt_obj = os.path.join(self.dataset_dir, 'ImageSets', 'train_obj.txt' if self.mode == 'train' else 'val_obj.txt')
        self.sample_id_obj = [x.strip() for x in open(self.split_txt_obj).readlines()]

        # Split for lanes
        if self.mode == 'train':
            with open(os.path.join(self.dataset_dir, 'ImageSets', 'train_gt.txt'), 'r') as f:
                self.list_lanes = f.readlines()
                self.list_lanes = self.list_lanes[150:400]
        elif self.mode == 'val':
            with open(os.path.join(self.dataset_dir, 'ImageSets', 'val_lanes.txt'), 'r') as f:
                self.list_lanes = f.readlines()
                # self.list_lanes = self.list_lanes[:150]

        # Transformations
        self.img_transform = transforms.Compose([transforms.Resize((self.input_height, self.input_width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

        self.segment_transform = transforms.Compose([FreeScaleMask((self.lane_output_height, self.lane_output_width)),
                                                     MaskToTensor(), ])

        self.num_samples = len(self.list_lanes)

        print('Loaded %d self labeled samples for %s' % (self.num_samples, self.mode))

    def __getitem__(self, index):

        # Get paths
        img_id = self.list_lanes[index].split()[0]
        label_id = self.list_lanes[index].split()[1]

        img_path = os.path.join(self.img_path, img_id)
        label_path = os.path.join(self.dataset_dir, 'training', label_id)


        img = Image.open(img_path)
        label = Image.open(label_path)

        width, height = img.size

        annotations = os.path.join(self.annot_path, img_id[:-4] + '.txt')
        cls, boxes = self.get_obj_labels(annotations)

        if self.mode == 'train':
            transform = Aug.Compose([Aug.ShiftScaleRotate(
                                         scale_limit=[0., 0.],
                                         shift_limit=0.3,
                                         rotate_limit=0.0,
                                         border_mode=0,
                                         interpolation=cv2.INTER_LINEAR,
                                         p=1)],
                                    bbox_params=Aug.BboxParams(format='pascal_voc',
                                                               label_fields=['category_ids']),
                                    additional_targets={'label': 'image'},)

            transformed = transform(image=np.array(img), bboxes=boxes, category_ids=cls, label=np.array(label))
            img, label = Image.fromarray(transformed['image']), Image.fromarray(transformed['label'])
            boxes, cls = transformed['bboxes'], np.array(transformed['category_ids'])

        if len(boxes) == 0:
            boxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            cls = np.array([[0]])

        test_img = np.array(img)
        test_img = cv2.resize(test_img, (256, 144))


        # wandb.log({"examples":[wandb.Image(test_img, caption = "augmentation_check")]})

        img = self.img_transform(img)

        lane_pts = self._get_index(label)

        cls_label = self._grid_pts(lane_pts, self.griding_num, width)

        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        hmap = np.zeros((self.num_classes, self.obj_output_height, self.obj_output_width), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        for k, (bbox, label) in enumerate(zip(boxes, cls)):
            x1, y1, x2, y2 = [elem for elem in bbox]

            x1, x2 = x1*self.input_width/float(width)/4., x2*self.input_width/float(width)/ 4.
            y1, y2 = y1 * self.input_height/float(height)/4., y2*self.input_height/float(height)/4.
            for bbox in boxes:
                test_img = cv2.rectangle(test_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            h, w = y2 - y1, x2 - x1
            if h > 0 and w > 0:
                obj_c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                hmap[label] = draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.obj_output_width + obj_c_int[0]
                ind_masks[k] = 1


        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'cls_label':cls_label, 'seg_label':seg_label, 'test_img':test_img}  # 'img_id': img_id}

    def __len__(self):
        return self.num_samples

    def get_obj_labels(self,annotation):
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
            bbox = np.array(
                [float(line_parts[1]), float(line_parts[2]), float(line_parts[3]), float(line_parts[4])])
            labels.append(cat_id)
            boxes.append(bbox)

        return np.asarray(labels), np.asarray(boxes)

    def _grid_pts(self, pts, num_cols, w):
        """
        pts: indexes of lane positions
        num_cols: number of columns to look for
        w: width of input image
        returns coordinates in image frame
        """
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        """
        label: label image (segmentations mask)
        returns indexes of lane coordinates
        """
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
