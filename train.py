import argparse
import torch
import numpy as np
import random
from data_loader.data_loaders import ObjectDataLoader, LanesDataLoader, LabeledDataLoader
import wandb

from model.model import AegisMTModel
from model.backbone import resnet
from model.heads import AegisLaneHead, AegisObjHead
from parse_config import ConfigParser
from trainer import Trainer2Datasets, Trainer2DStochastic, Trainer
from utils import cp_projects
from evaluation.run_evaluation import run_eval

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)

def main(config):

    # Create a logger (saves training info in)
    logger = config.get_logger('train')
    # setup data_loader instances
    if config['datasets']['obj_and_lanes'] == True:
        train_data_loader = LabeledDataLoader(config, 'train')
        val_data_loader = LabeledDataLoader(config, 'val')
        config['arch']['cls_num_per_lane'] = 56
        config['arch']['griding_num'] = 100
        config['loss']['sim_loss_w'] = 1.0
    else:
        train_data_loaders = {'objects': ObjectDataLoader(config, 'train'),
                              'lanes': LanesDataLoader(config, 'train')}
        val_data_loaders = {'objects': ObjectDataLoader(config, 'val'),
                            'lanes': LanesDataLoader(config, 'val')}

        config['arch']['cls_num_per_lane'] = 56 if config['datasets']['lanes'] == 'tusimple' else 18
        config['arch']['griding_num'] = 100 if config['datasets']['lanes'] == 'tusimple' else 200
        config['loss']['sim_loss_w'] = 1.0 if config['datasets']['lanes'] == 'tusimple' else 0.0

    # build model architecture, then print to console
    ## define the shared backbone
    backbone = resnet(config)
    config['arch']['inplanes'] = backbone.inplanes
    ## define all individual heads
    lane_head = AegisLaneHead(config)
    obj_head = AegisObjHead(config)

    # Track training with wandb
    wandb.init(config=config, group=config['tasks'], name=config['run'], save_code=True)

    model = AegisMTModel(bbone=backbone, head_lane=lane_head, head_obj=obj_head)

    wandb.watch(model)

    logger.info(model)

    # prepare for GPU training
    model = model.to(config['device'])

    # build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if config['datasets']['lanes'] == 'tusimple':
        optimizer = torch.optim.Adam(trainable_params,
                                     lr=config['optimizer']['args']['lr'],
                                     weight_decay=config['optimizer']['args']['weight_decay'],
                                     amsgrad=config['optimizer']['args']['amsgrad'])


        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config['lr_scheduler']['args']['step_size'],
                                                    gamma=config['lr_scheduler']['args']['gamma'])
    else:
        optimizer = torch.optim.SGD(trainable_params,
                                    lr=0.025,
                                    momentum=0.9,
                                    weight_decay=0.0001)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=15,
                                                    gamma=0.1)

    if config['datasets']['obj_and_lanes'] == True:
        trainer = Trainer(model,
                          train_data_loader, val_data_loader,
                          optimizer, scheduler,
                          config)

    # Define trainer
    else:
        if config['trainer']['stochastic'] == True:
            trainer = Trainer2DStochastic(model,
                                          train_data_loaders['lanes'], val_data_loaders['lanes'],
                                          train_data_loaders['objects'], val_data_loaders['objects'],
                                          optimizer, scheduler,
                                          config)
        else:
            trainer = Trainer2Datasets(model,
                                       train_data_loaders['lanes'], val_data_loaders['lanes'],
                                       train_data_loaders['objects'], val_data_loaders['objects'],
                                       optimizer, scheduler,
                                       config)

    # Run trainer unless specifically stated to evaluate only
    if not config['trainer']['eval_only']:
        trainer.train()
    run_eval(config)

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-m', '--mode', default='train', type=str,
                      help='model can be train or test')

    # Update config file with arguments provided by user
    config = ConfigParser.from_args(args)

    # Create a copy of all files
    cp_projects(config)

    main(config)
