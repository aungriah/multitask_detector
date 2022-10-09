import os, torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from dataset.utils.obj_transformations import ctdet_decode

def gt_and_detections(config,model,dataset):
    """
    config: hyperparameters of trained model
    model: network to be evaluated
    dataset: dataset used to perform evaluation on
    output: Writes predictions and annotations files to directories for further comparison
    """

    det_dir = os.path.join(config._save_dir, 'evaluation', 'objects', 'detection-results')
    gt_dir = os.path.join(config._save_dir, 'evaluation', 'objects', 'ground-truth')
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    VALNAME_2_NAME = {
        'Pedestrian': 'Pedestrian',
        'pedestrian': 'Pedestrian',
        'person': 'Pedestrian',
        'Car': 'Car',
        'Van': 'Car',
        'car': 'Car',
        'trailer': 'Car',
        'Cyclist': 'Rider',
        'bicycle': 'Rider',
        'motorcylce': 'Rider',
        'rider': 'Rider',
        'Truck': 'XLVehicle',
        'bus': 'XLVehicle',
        'truck': 'XLVehicle',
        'Tram': 'Tram',
        'train': 'Tram',
        'Person_sitting': 'ignore',
        'Misc': 'ignore',
        'DontCare': 'ignore'

    }

    MAP_2_NAME = {
        1: 'Pedestrian',
        2: 'Car',
        3: 'Rider',
        4: 'XLVehicle',
        5: 'Tram'
    }

    obj_dataloader_name = dataset + '_dataloader'
    test_images_dir = os.path.join(config[obj_dataloader_name]['args']['data_dir'])
    img_list_dir = os.path.join(test_images_dir, 'ImageSets', 'eval.txt')
    if dataset == 'finetuning':
        img_list_dir = os.path.join(test_images_dir, 'test.txt')
    img_list = [x.strip() for x in open(img_list_dir).readlines()]

    # Prepare predictions -------------------------------------------------------------------------------------------- #

    img_transform = transforms.Compose([
        transforms.Resize((config['arch']['input_height'], config['arch']['input_width'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    for name in img_list:

        if dataset == 'finetuning':
            img_path = os.path.join(test_images_dir, name)
            image_name = name.split('/')[-1]
            filename = det_dir + '/' + image_name[:-3] + 'txt'
        elif dataset == 'kitti':
            img_path = os.path.join(test_images_dir, 'training', 'image', '{:06d}.png'.format(int(name)))
            filename = det_dir + '/' + '{:06d}.txt'.format(int(name))
        else:
            img_path = os.path.join(test_images_dir, 'training', 'image', '{}.jpg'.format(name))
            filename = det_dir + '/' + '{}.txt'.format(name)

        image = Image.open(img_path)
        width, height = image.size
        image = img_transform(image)[None, :, :, :]

        with torch.no_grad():
            detections = []
            image = image.to('cuda')

            output_obj, output_lanes = model(image)
            output = output_obj[-1]
            dets = ctdet_decode(*output, K=100)
            dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

            top_preds = {}

            dets[:, 0] = dets[:, 0] * 4. * width / config['arch']['input_width']
            dets[:, 1] = dets[:, 1] * 4. * height / config['arch']['input_height']
            dets[:, 2] = dets[:, 2] * 4. * width / config['arch']['input_width']
            dets[:, 3] = dets[:, 3] * 4. * height / config['arch']['input_height']

            cls = dets[:, -1]
            for j in range(config['arch']['num_obj_classes']):
                inds = (cls == j)
                top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                top_preds[j + 1][:, :4] /= 1

            detections.append(top_preds)

            bbox_and_scores = {}
            for j in range(1, config['arch']['num_obj_classes']+1):
                bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, config['arch']['num_obj_classes']+1)])

            if len(scores) > 100:
                kth = len(scores) - 100
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, config['arch']['num_obj_classes']+1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            with open(filename, 'w') as f:
                for lab in bbox_and_scores:
                    for boxes in bbox_and_scores[lab]:
                        x1, y1, x2, y2, score = boxes
                        if score > 0.25:
                            laneToWrite = MAP_2_NAME[lab] + ' ' + str(score) + ' ' + str(x1) + ' ' + str(
                                y1) + ' ' + str(x2) + ' ' + str(y2) + '\n'
                            f.writelines(laneToWrite)
            f.close()

    # Prepare targets ------------------------------------------------------------------------------------------------ *

    for name in img_list:

        if dataset == 'finetuning':
            image_name = name.split('/')[-1]
            label_path = os.path.join(test_images_dir, 'eval', 'bboxes', image_name[:-3] + 'txt')
            gt_filename = gt_dir + '/' + image_name[:-3] + 'txt'
            adjust = 3
        elif dataset == 'kitti':
            label_path = os.path.join(test_images_dir, 'training', 'label', '{:06d}.txt'.format(int(name)))
            gt_filename = gt_dir + '/' + '{:06d}.txt'.format(int(name))
            adjust = 0
        else:
            label_path = os.path.join(test_images_dir, 'training', 'label', '{}.txt'.format(name))
            gt_filename = gt_dir + '/' + '{}.txt'.format(name)
            adjust = 3

        with open(gt_filename, 'w') as f:
            for line in open(label_path, 'r'):
                line = line.rstrip()
                line_parts = line.split(' ')
                class_name = line_parts[0]
                cat = VALNAME_2_NAME[class_name]
                if cat == 'ignore':
                    continue
                x1 = line_parts[4-adjust]
                y1 = line_parts[5-adjust]
                x2 = line_parts[6-adjust]
                y2 = line_parts[7-adjust]

                lineToWrite = cat + ' ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + '\n'
                f.writelines(lineToWrite)
        f.close()

