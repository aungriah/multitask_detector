"""
This script is used to draw bounding boxes around objects.
It serves as test to see weather annotations are correct
"""
import numpy as np
import os, cv2
from PIL import Image
import albumentations as Aug
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

data_dir = '/Users/aungriah/Desktop/ownImages'
images = os.path.join(data_dir, 'image')
labels = os.path.join(data_dir, 'label')
selected = '/Users/aungriah/Desktop/ownImages/ImageSets/train.txt'

CLASS_NAME_TO_ID = {
    'person':0,
    'car':1,
    'bus':1,
    'bicycle':3,
    'motorcycle':3,
    'rider':3,
    'train':4,
    'truck':1,
    'trailer':5
}

idx = []
for line in open(selected,'r'):
    line=line.rstrip()
    idx.append(line)
print(len(idx))

def get_labels(annotation):
    labels = []
    boxes = []
    for line in open(annotation, 'r'):
        line = line.rstrip()
        line_parts = line.split(' ')
        obj_name = line_parts[0]
        cat_id=obj_name
        bbox = np.array([float(line_parts[1]), float(line_parts[2]), float(line_parts[3]), float(line_parts[4])])
        labels.append(cat_id)
        boxes.append(bbox)

    return np.asarray(labels), np.asarray(boxes)

img_folder = os.listdir(images)
img_folder.sort()
for file in idx:

    label_file = labels + '/' + file + '.txt'

    img = Image.open(os.path.join(images, file + '.png'))

    width, height = img.size
    print(label_file, file)

    label, boxes = get_labels(label_file)
    transform = Aug.Compose( [
                            Aug.HorizontalFlip(p=1),
                            Aug.MotionBlur(blur_limit=[7,13], p=1),
                            Aug.RandomBrightness(limit=0.2, p=1),
                            Aug.ShiftScaleRotate(scale_limit=0.3,
                                               shift_limit_x=0.3,
                                               shift_limit_y=0.4,
                                               rotate_limit=0,
                                               border_mode = 0,
                                               p=1)],
                            bbox_params = Aug.BboxParams(format='pascal_voc', label_fields=['category_ids']),)

    transformed = transform(image=np.array(img), bboxes=boxes, category_ids=label)
    new_image, new_boxes, new_label = Image.fromarray(transformed['image']), transformed['bboxes'], np.array(transformed['category_ids'])

    new_image = new_image.resize((1280,384))
    new_image = new_image.resize((320,96))
    print(new_label)

    f = plt.figure()

    ax = f.add_subplot(1, 2, 1)
    plt.imshow(img)
    for bbox in boxes:
        x1, y1, x2, y2 = bbox

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    ax2 = f.add_subplot(1, 2, 2)
    plt.imshow(new_image)
    for bbox in new_boxes:
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * 1280. / float(width) / 4., x2 * 1280. / float(width) / 4.
        y1, y2 = y1 * 384. / float(height) / 4., y2 * 384. / float(height) / 4.

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
    plt.show()

