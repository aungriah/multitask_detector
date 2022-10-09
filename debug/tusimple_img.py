"""
This scriot is used to draw lanes on annotated images
It serves as test to see weather annotations are correct
"""
import numpy as np
import os, cv2
from PIL import Image
import albumentations as Aug
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NAME_TO_ID = {
    'car':1,
    'truck':3,
    'pedestrian':0,
    'cyclist':2,
    'rider':2,
    'tram':4
}
def get_labels(annotation):
    labels = []
    boxes = []
    for line in open(annotation, 'r'):
        line = line.rstrip()
        line_parts = line.split(' ')
        obj_name = line_parts[0]
        # cat_id = CLASS_NAME_TO_ID[obj_name]
        # if cat_id <= -1:  # ignore Tram and Misc
        #     continue
        cat_id=obj_name
        bbox = np.array([float(line_parts[1]), float(line_parts[2]), float(line_parts[3]), float(line_parts[4])])
        labels.append(cat_id)
        boxes.append(bbox)

    return np.asarray(labels), np.asarray(boxes)

data_dir = '/Users/aungriah/Desktop/Finetuning/finetuning/training'

label = Image.open(os.path.join(data_dir,'seglabel', 'image_402.png'))
img = Image.open(os.path.join(data_dir,'image', 'image_402.jpg'))
width, height = img.size
annot = '/Users/aungriah/Desktop/Finetuning/finetuning/training/label/image_402.txt'

cv2_img = np.array(img)

labels, boxes = get_labels(annot)

transform = Aug.Compose( [Aug.ShiftScaleRotate(scale_limit=[0.0,0.0],
                                               shift_limit_x=0.3,
                                               shift_limit_y=0.2,
                                               rotate_limit=7,
                                               border_mode = 0,
                                               p=1)],
                            bbox_params=Aug.BboxParams(format='pascal_voc',label_fields=['category_ids']),
                            additional_targets={"label":"image"},)

transformed = transform(image=np.array(img), label=np.array(label), bboxes=boxes, category_ids=labels)
new_image, new_label = Image.fromarray(transformed['image']), Image.fromarray(transformed['label'])
new_boxes = transformed['bboxes']

f = plt.figure()
f.add_subplot(2,2,1)
plt.imshow(img)
f.add_subplot(2,2,2)
plt.imshow(label)
ax3 = f.add_subplot(2,2,3)
plt.imshow(new_image)
for bbox in new_boxes:
    x1, y1, x2, y2 = bbox

    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
    ax3.add_patch(rect)
f.add_subplot(2,2,4)
plt.imshow(new_label)

plt.show()

