import torch, argparse, os, cv2, scipy.special
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np

from model.model import AegisMTModel
from model.backbone import resnet
from model.heads import AegisLaneHead, AegisObjHead
from dataset.utils.obj_transformations import ctdet_decode
from parse_config import ConfigParser
from dataset.utils.lane_anchors import tusimple_row_anchor

color = {0: (225,0,0),
         1: (0,255,0),
         2: (0,0,255),
         3: (255,0,255)
         }

def main(config):

    row_anchor = tusimple_row_anchor

    img_transform = transforms.Compose([
        transforms.Resize((config['arch']['input_height'], config['arch']['input_width'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # build model architecture
    backbone = resnet(config)
    config['arch']['cls_num_per_lane'] = 56 if config['datasets']['lanes'] == 'tusimple' else 18
    config['arch']['griding_num'] = 100 if config['datasets']['lanes'] == 'tusimple' else 200
    config['arch']['inplanes'] = backbone.inplanes
    config['arch']['use_aux'] = False

    ## define all individual heads
    lane_head = AegisLaneHead(config)
    obj_head = AegisObjHead(config)

    ## define model
    model = AegisMTModel(bbone=backbone, head_lane=lane_head, head_obj=obj_head)
    model.to(config['device'])

    ## load model weights
    checkpoint = torch.load(config['trainer']['path_to_weights'])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Read images and create directory to store results
    path_to_images = config['image_dir']
    img_dir = os.listdir(path_to_images)
    img_dir = sorted(img_dir)
    path_to_save_dir = config['save_dir']

    # Uncomment to create mp4 video
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #vout = cv2.VideoWriter('test_small.avi', fourcc, 30.0, (1280, 720))
    with torch.no_grad():
        for img_idx, file in enumerate(tqdm(img_dir)):

            name = file[:-4]
            file_ext = file[len(name):]
            print(name, file_ext)
            img = Image.open(os.path.join(path_to_images, file))
            width, height = img.size
            img_to_save = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            img = img_transform(img)[None, :, :, :]
            img = img.to(config['device'])

            detections = []
            output_obj, output_lanes = model(img)
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

            for lab in bbox_and_scores:
                for boxes in bbox_and_scores[lab]:
                    x1, y1, x2, y2, score = boxes
                    if score > 0.25:
                        img_to_save = cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (255, 0, 0), 2)

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

            left = []
            right = []
            for i in range(1,3):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * width / 800) -8,#25*(i==1)*row_anchor[config['arch']['cls_num_per_lane'] - 1 - k] / 288,
                                   int(height * (row_anchor[config['arch']['cls_num_per_lane'] - 1 - k] / 288)) - 1)
                            left.append(ppp) if i==1 else right.append(ppp)

                            # Uncomment to plot outer lanes
                            img_to_save = cv2.circle(img_to_save, ppp, 5, (0,0,255), -1)

            # # Uncomment for plotting center line
            # for k in range(out_j.shape[0]):
            #     if out_j[k, 1] > 0 and out_j[k, 2] > 0:
            #         ppp1 = (int(out_j[k, 1] * col_sample_w * width / 800) - 8,
            #                # 25*(i==1)*row_anchor[config['arch']['cls_num_per_lane'] - 1 - k] / 288,
            #                int(height * (row_anchor[config['arch']['cls_num_per_lane'] - 1 - k] / 288)) - 1)
            #         ppp2 = (int(out_j[k, 2] * col_sample_w * width / 800) - 8,
            #                # 25*(i==1)*row_anchor[config['arch']['cls_num_per_lane'] - 1 - k] / 288,
            #                int(height * (row_anchor[config['arch']['cls_num_per_lane'] - 1 - k] / 288)) - 1)
            #         ppp = ((ppp1[0]+ppp2[0])//2, (ppp1[1]+ppp2[1])//2)
            #         img_to_save = cv2.circle(img_to_save, ppp, 5, (0, 255, 0), -1)

            # Uncomment to plot the drivable are
            # if (len(right) > 0 and len(left)  > 0):
            #     right = right[::-1]
            #     window_img = np.zeros_like(img_to_save)
            #     left = np.array(left, dtype=np.int32)
            #     right = np.array(right, dtype=np.int32)
            #     left = left.reshape(1,-1,2)
            #     right = right.reshape(1,-1,2)
            #     seg_area = np.concatenate((left, right), axis=1)
            #     window_img = cv2.fillPoly(window_img, seg_area, (0, 255, 0))
            #     img_to_save = cv2.addWeighted(img_to_save, 1, window_img, 0.3, 0)

            cv2.imwrite(os.path.join(path_to_save_dir, name + 'prediction' + file_ext), img_to_save)

            # Uncomment 2 lines below to write to mp4 video
            #vout.write(img_to_save)
        #vout.release()

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-m', '--mode', default='test', type=str,
                      help='train or test (default: test)')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-p2w', '--weights', required=True,
                      help='path to weights (default: None)')
    args.add_argument('-p2i', '--imgdir', required=True,
                      help='path to image directory (default: None)')
    args.add_argument('-s', '--save', required=True,
                      help='path to directory to save images (default: None)')
    args.add_argument('-d', '--device', default='cuda', type=str,
                      help='indices of GPUs to enable (default: all)')


    config = ConfigParser.from_args(args)

    main(config)
