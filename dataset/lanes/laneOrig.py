import torch
from PIL import Image
import os
import pdb
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import albumentations as Aug

from dataset.utils.augmentations import find_start_pos, RandomRotate, RandomLROffsetLABEL, RandomUDoffsetLABEL, Compose2, FreeScaleMask, MaskToTensor
from dataset.utils.lane_anchors import tusimple_row_anchor, culane_row_anchor

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)

def loader_func(path):
    return Image.open(path)


class LaneDataset(Dataset):
    def __init__(self, config, mode):
        super(LaneDataset, self).__init__()

        self.config = config
        self.specific_dataset = config['datasets']['lanes'] + '_dataloader'

        self.griding_num = config[self.specific_dataset]['args']['griding_num']
        self.num_lanes = config[self.specific_dataset]['args']['num_lanes']
        self.use_aux = config['arch']['use_aux']
        self.dataset_dir = config[self.specific_dataset]['args']['data_dir']
        self.mode = mode
        self.use_aux = config['arch']['use_aux']
        self.row_anchor = tusimple_row_anchor if config['datasets']['lanes'] == 'tusimple' else culane_row_anchor
        self.row_anchor.sort()

        self.input_width =self.config['arch']['input_width']
        self.input_height = self.config['arch']['input_height']
        self.output_width =  self.input_width//8
        self.output_height = self.input_height//8

        # --------------------------- Transformations to be applied --------------------------- #
        self.img_transform = transforms.Compose([transforms.Resize((self.input_height, self.input_width)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        self.simu_transform = Compose2([RandomRotate(7),
                                        RandomUDoffsetLABEL(100),
                                        RandomLROffsetLABEL(200),])
        self.segment_transform = transforms.Compose([FreeScaleMask((self.output_height, self.output_width)),
                                                     MaskToTensor(),])

        if self.mode == 'train':
            with open(os.path.join(self.dataset_dir, 'train.txt'), 'r') as f:
                self.list = f.readlines()
        elif self.mode == 'val':
            with open(os.path.join(self.dataset_dir, 'val.txt'), 'r') as f:
                self.list = f.readlines()
                if config['datasets']['lanes'] == 'culane':
                    random.shuffle(self.list)
                    self.list = self.list[:1500]
        else:
            with open(os.path.join(self.dataset_dir, 'test.txt'), 'r') as f:
                self.list = f.readlines()
            self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane

        print('Loaded {} {} lane images!'.format(len(self.list), self.mode))

    def __getitem__(self, index):

        if self.mode == 'test':

            name = self.list[index].split()[0]
            img_path = os.path.join(self.dataset_dir, name)
            img = loader_func(img_path)

            if self.img_transform is not None:
                img = self.img_transform(img)

            return img, name

        else:

            l = self.list[index]
            l_info = l.split()
            img_name, label_name = l_info[0], l_info[1]
            if img_name[0] == '/':
                img_name = img_name[1:]
                label_name = label_name[1:]

            label_path = os.path.join(self.dataset_dir, label_name)
            label = loader_func(label_path)

            img_path = os.path.join(self.dataset_dir, img_name)
            img = loader_func(img_path)

            if self.simu_transform is not None and self.mode == 'train':
                img, label = self.simu_transform(img, label)

            # get indexes where lanes are located
            lane_pts = self._get_index(label)

            # get the coordinates of lanes at row anchors
            w, h = img.size
            cls_label = self._grid_pts(lane_pts, self.griding_num, w)

            # make the coordinates to classification label
            if self.use_aux:
                assert self.segment_transform is not None
                seg_label = self.segment_transform(label)

            if self.img_transform is not None:
                img = self.img_transform(img)

            if self.use_aux:
                return img, cls_label, seg_label

            return img, cls_label

    def __len__(self):
        return len(self.list)

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