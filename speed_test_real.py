import torch, argparse, os, cv2, scipy.special, time
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np

from model.model import AegisMTModel
from model.backbone import resnet
from model.heads import AegisLaneHead, AegisObjHead
from parse_config import ConfigParser
from dataset.utils.obj_transformations import ctdet_decode

from dataset.utils.lane_anchors import tusimple_row_anchor

torch.backends.cudnn.benchmark = True

def main(config):
    img_transform = transforms.Compose([
        transforms.Resize((config['arch']['input_height'], config['arch']['input_width'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    backbone = resnet(config)
    config['arch']['cls_num_per_lane'] = 56
    config['arch']['griding_num'] = 100
    config['arch']['inplanes'] = backbone.inplanes
    config['arch']['use_aux'] = False

    ## define all individual heads
    lane_head = AegisLaneHead(config)
    obj_head = AegisObjHead(config)

    model = AegisMTModel(bbone=backbone, head_lane=lane_head, head_obj=obj_head)
    model.to('cuda')
    model.eval()



    warmup = 150

    for i in range(warmup):
        img = Image.open('sample.jpg')
        width, height = img.size
        img = img_transform(img)[None, :, :, :]
        img = img.to('cuda')

        with torch.no_grad():
            output_obj, output_lanes = model(img)

    img1 = cv2.imread('sample.jpg', cv2.COLOR_BGR2RGB)

    height, width = 720, 1280
    t_all = []
    t_preprocess = []
    t_forward_pass = []
    t_postprocess = []
    for i in range(250):

        # Preprocessing
        t1 = time.time()
        # img = Image.open('sample.jpg')
        img = img1.copy()
        # height, width = img1.shape[:2]
        # width, height = img.size
        # img = cv2.resize(img, (config['arch']['input_width'], config['arch']['input_height']))
        img = cv2.resize(img, (1024, 576))
        img = (img.astype(np.float32) / 255. - mean_rgb) / std_rgb

        img = torch.from_numpy(img.transpose(2,0,1))[None, :, :, :]
        img = img.to('cuda')
        t2 = time.time()

        # Forward Pass
        detections = []
        with torch.no_grad():
            output_obj, output_lanes = model(img)
        t3 = time.time()

        # Postprocess
        # -- Objects
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
        for j in range(1, config['arch']['num_obj_classes'] + 1):
            bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
        scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, config['arch']['num_obj_classes'] + 1)])

        if len(scores) > 100:
            kth = len(scores) - 100
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, config['arch']['num_obj_classes'] + 1):
                keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

        # -- Lanes
        col_sample = np.linspace(0, 800 - 1, config['arch']['griding_num'])
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = output_lanes[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(config['arch']['griding_num']) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == config['arch']['griding_num']] = 0
        out_j = loc

        t4 = time.time()
        t_preprocess.append(t2-t1)
        t_forward_pass.append(t3-t2)
        t_postprocess.append(t4-t3)
        t_all.append(t4 - t1)

    print('------------------------------------------------------------------------------------------------------------')
    print('Inference average fps:', 1 / np.mean(t_all))
    print('Preprocessing average fps:', 1 / np.mean(t_preprocess))
    print('Forward pass average fps:', 1 / np.mean(t_forward_pass))
    print('Postprocessing average fps:', 1 / np.mean(t_postprocess))
    print('------------------------------------------------------------------------------------------------------------')

    print('Time statistics of whole pipeline (Pre-Processing, Forward Pass, Post-processing')
    print('--- average time:', np.mean(t_all) / 1)
    print('---  average fps:',1 / np.mean(t_all))

    print('--- fastest time:', min(t_all) / 1)
    print('---  fastest fps:',1 / min(t_all))

    print('--- slowest time:', max(t_all) / 1)
    print('---  slowest fps:',1 / max(t_all))

    print('Time statistics for Pre-Processing')
    print('--- average time:', np.mean(t_preprocess) / 1)
    print('---  average fps:', 1 / np.mean(t_preprocess))

    print('--- fastest time:', min(t_preprocess) / 1)
    print('---  fastest fps:', 1 / min(t_preprocess))

    print('--- slowest time:', max(t_preprocess) / 1)
    print('---  slowest fps:', 1 / max(t_preprocess))

    print('Time statistics for forward pass')
    print('--- average time:', np.mean(t_forward_pass) / 1)
    print('---  average fps:', 1 / np.mean(t_forward_pass))

    print('--- fastest time:', min(t_forward_pass) / 1)
    print('---  fastest fps:', 1 / min(t_forward_pass))

    print('--- slowest time:', max(t_forward_pass) / 1)
    print('---  slowest fps:', 1 / max(t_forward_pass))

    print('Time statistics for Post-Processing')
    print('--- average time:', np.mean(t_postprocess) / 1)
    print('---  average fps:', 1 / np.mean(t_postprocess))

    print('--- fastest time:', min(t_postprocess) / 1)
    print('---  fastest fps:', 1 / min(t_postprocess))

    print('--- slowest time:', max(t_postprocess) / 1)
    print('---  slowest fps:', 1 / max(t_postprocess))

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default='config.json', type=str)
    args.add_argument('-m', '--mode', default='speed_test', type=str)
    args.add_argument('-bb', '--backbone', default=34, type=str)
    args.add_argument('-w_', '--width', default=1024, type=str)
    args.add_argument('-h_', '--height', default=576, type=str)

    config = ConfigParser.from_args(args)

    main(config)