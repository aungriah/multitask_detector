import os, torch, shutil

from evaluation.lanes.evaluate import eval_lanes
from evaluation.lanes.evaluate_ft import eval_lanes_ft
from evaluation.objects.evaluate import eval_objects
from evaluation.objects.prepareData import gt_and_detections

from model.model import AegisMTModel
from model.backbone import resnet
from model.heads import AegisLaneHead, AegisObjHead

def run_eval(config):
    """
    config: config file indicating hyperparameters of the model
    runs evaluation on object detection and lane detection task
    """

    logger = config.get_logger('trainer', 2)

    logger.info('Loading best model for evaluation!')
    print('Loading best model for evaluation!')
    # build model architecture
    backbone = resnet(config)
    config['arch']['cls_num_per_lane'] = 56 if config['datasets']['lanes'] == 'tusimple' else 18
    config['arch']['griding_num'] = 100 if config['datasets']['lanes'] == 'tusimple' else 200
    config['arch']['inplanes'] = backbone.inplanes
    config['arch']['use_aux'] = False
    ## define all individual heads
    lane_head = AegisLaneHead(config)
    obj_head = AegisObjHead(config)

    model = AegisMTModel(bbone=backbone, head_lane=lane_head, head_obj=obj_head)
    model.to(config['device'])

    if config['datasets']['obj_and_lanes'] == False:
        if os.path.exists(os.path.join(config['trainer']['checkpoint_dir'], 'model_best.pth')):
            model_weights = os.path.join(config['trainer']['checkpoint_dir'], 'model_best.pth')

    elif config['datasets']['obj_and_lanes'] == True:
        if os.path.exists(os.path.join(config['trainer']['checkpoint_dir'], 'best_train_model.pth')):
            model_weights = os.path.join(config['trainer']['checkpoint_dir'], 'best_train_model.pth')
    else:
        model_weights = config['trainer']['path_to_weights']

    checkpoint = torch.load(model_weights)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print('Model loaded')

    save_dir = os.path.join(config._save_dir, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)


    # Evaluate ------------------------------------------------------------------------------------------------------- #
    logger.info('Running evaluation on lane detector')

    if config['datasets']['obj_and_lanes'] == False:
        print('Running evaluation on TuSimple lane detector')
        eval_lanes(config, model)

        print('Preparing targets for bdd100k')
        gt_and_detections(config, model, 'bdd100k')
        print('Running evaluation on bdd100k with IoU=0.5')
        logger.info('Running evaluation on bdd100k with IoU=0.5')
        eval_objects(config, 'bdd100k', min_overlap=0.5)

        logger.info('Removing bdd100k evaluation files')
        print('Removing bdd100k evaluation files')
        shutil.rmtree(os.path.join(config._save_dir, 'evaluation', 'objects', 'detection-results'))
        shutil.rmtree(os.path.join(config._save_dir, 'evaluation', 'objects', 'ground-truth'))
    else:


        print('Preparing object targets os self-labeled images')
        gt_and_detections(config, model, 'finetuning')
        print('Preparing object targets os self-labeled images')
        logger.info('Running evaluation on bdd100k with IoU=0.5')
        eval_objects(config, 'finetuning', min_overlap=0.5)

        logger.info('Removing self-labeled evaluation files')
        print('Removing self-labeled evaluation files')
        shutil.rmtree(os.path.join(config._save_dir, 'evaluation', 'objects', 'detection-results'))
        shutil.rmtree(os.path.join(config._save_dir, 'evaluation', 'objects', 'ground-truth'))

        print('Running evaluation on self-labeled lane detector')
        eval_lanes_ft(config, model)

    # print('Preparing targets for kitti')
    # gt_and_detections(config, model, 'kitti')
    # print('Running evaluation on kitti with IoU=0.5')
    # logger.info('Running evaluation on kitti with IoU=0.5')
    # eval_objects(config, 'kitti',min_overlap=0.5)
    # shutil.rmtree(os.path.join(config._save_dir, 'evaluation', 'objects', 'detection-results'))
    # shutil.rmtree(os.path.join(config._save_dir, 'evaluation', 'objects', 'ground-truth'))


