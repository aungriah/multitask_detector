from PIL import Image, ImageOps, ImageFilter
from skimage import transform as trans
import imgaug.augmenters as iaa
import cv2, random

import numpy as np
import torchvision.transforms as transforms
import torch

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)

# PIL affine transformations ----------------------------------------------------------------------------

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    np.random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


# ===============================img tranforms============================

class Compose2(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


class FreeScale(object):
    def __init__(self, size):
        """
        size: desired dimensions
        returns image and mask resized to size
        """
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]),
                                                                                     Image.NEAREST)


class FreeScaleMask(object):
    def __init__(self, size):
        self.size = size
        """
        size: desired dimensions
        returns mask resized to size
        """
    def __call__(self, mask):
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomRotate(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, angle):
        """
        angle: angle at which image and label should be rotated
        returns rotated image and label
        """
        self.angle = angle

    def __call__(self, image, label):
        assert label is None or image.size == label.size

        ### Random angle
        random_angle = np.random.randint(0, 90)
        label = label.rotate(random_angle, resample=Image.NEAREST)
        image = image.rotate(random_angle, resample=Image.BILINEAR)

        angle = np.random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        label = label.rotate(-1*random_angle, resample=Image.NEAREST)
        image = image.rotate(-1*random_angle, resample=Image.BILINEAR)

        return image, label


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        """
        mean: mean per channel
        std: standard deviation per channel
        return denormalized image
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    """
    img: image
    returns image as tensor
    """
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def find_start_pos(row_sample, start_line):
    # row_sample = row_sample.sort()
    # for i,r in enumerate(row_sample):
    #     if r >= start_line:
    #         return i
    l, r = 0, len(row_sample) - 1
    while True:
        mid = int((l + r) / 2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid


class RandomLROffsetLABEL(object):
    """
    max_offset: maximum offset to be considered for translation in x coordinate
    returns randomly shifted image and label
    """
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        return Image.fromarray(img), Image.fromarray(label)


class RandomUDoffsetLABEL(object):
    """
    max_offset: maximum offset to be considered for translation in y coorrdinate
    returns randomly shifted image and label
    """
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        return Image.fromarray(img), Image.fromarray(label)

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import cv2


    def get_label(label_file):

        boxes = []
        for line in open(label_file, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]
            cat_id = CLASS_NAME_TO_ID[obj_name]
            if cat_id <= -1:  # ignore Tram and Misc
                continue
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
            boxes.append(bbox)
        return np.array(boxes)

    CLASS_NAME_TO_ID = {
        'Pedestrian': 0,
        'Car': 1,
        'Cyclist': 2,
        'Van': -2,
        'Truck': -3,
        'Motorcycle': -4,
        'Person_sitting': -5,
        'Tram': -6,
        'Misc': -7,
        'DontCare': -1
    }

    img_dir = '/Users/aungriah/Documents/MT/Data/kitti/training/image_2'
    label_dir = '/Users/aungriah/Documents/MT/Data/kitti/training/label_2'
    img_folder = os.listdir(img_dir)
    img_folder.sort()
    for file in img_folder:

        label_file = label_dir + '/' + file.split('.')[0] + '.txt'

        img = Image.open(os.path.join(img_dir, file))
        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        center[0] = size[0] - center[0] - 1

        boxes = get_label(label_file)
        for bbox in boxes:
            x1_old, x2_old = bbox[0], bbox[2]
            bbox[0] = size[0] - x2_old - 1
            bbox[2] = size[0] - x1_old - 1

        center[0] += size[0] * np.random.choice(np.arange(-0.3, 0.4, 0.1))
        center[1] += size[1] * np.random.choice(np.arange(-0.3, 0.4, 0.1))
        size *= np.random.choice(np.arange(0.7,1.1,0.1))






        input_width = 1280
        input_height = 384
        output_width = 320
        output_height = 96

        trans_affine = get_transfrom_matrix(
            center, size,
            [input_width, input_height]
        )

        trans_affine_inv = np.linalg.inv(trans_affine)

        img_trans = img.transform(
            (input_width, input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center, size,
            [output_width, output_height]
        )

        trans_mat_inv = np.linalg.inv(trans_mat)

        img_trans_mat = img.transform(
            (output_width, output_height),
            method=Image.AFFINE,
            data=trans_mat_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        for bbox in boxes:
            x1,y1,x2,y2 = bbox
            print(x2-x1, y2-y1)
            bbox[:2] = affine_transform(bbox[:2], trans_mat)
            bbox[2:] = affine_transform(bbox[2:], trans_mat)
            bbox[[0, 2]] = bbox[[0, 2]].clip(0, output_width - 1)
            bbox[[1, 3]] = bbox[[1, 3]].clip(0, output_height - 1)

        img_cv2 = np.array(img_trans_mat)

        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(img_trans)
        ax = f.add_subplot(1, 2, 2)
        plt.imshow(img_trans_mat)

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            print(x2-x1, y2-y1)

            rect = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            img_cv2=cv2.rectangle(img_cv2, (int(x1),int(y1)), (int(x2),int(y2)), color=(255,0,0), thickness = 1)
        plt.show()
        cv2.imshow('image', img_cv2)
        cv2.waitKey(10)
