import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from abc import abstractmethod
import os

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model,
                 train_data_loader, val_data_loader,
                 optimizer, scheduler,
                 config):
        """
        Inputs:
        model: defined architecture to be optimized
        train_data_loader: data_loader used for training
        val_data_loader: data_loader for validation
        optimizer: defined optimizer for training the network
        scheduler: defined scheduler for optimizer
        config: config file with all input configutation parameters
        """

        self.config = config
        self.logger = config.get_logger('trainer', 2)

        cfg_trainer = config['trainer']
        self.resume = cfg_trainer['resume']
        self.checkpoint_dir = cfg_trainer['checkpoint_dir']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.save_best = cfg_trainer['save_best']
        self.early_stop = cfg_trainer['early_stop']

        self.model = model

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.save_best:
            self.best_train = 100
            self.best_val = 100
            self.best_lane_value = 100
            self.best_obj_value = 100

        self.save_lane = False
        self.save_obj = False
        self.save_train = False

        self.start_epoch = 1

        if(self.resume):
            self._resume_checkpoint()

        # setup visualization writer instance                
        self.writer = SummaryWriter(os.path.join(cfg_trainer['log_dir'], 'tensorboard'))


    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            result_train = self._train_epoch(epoch)
            result_val = self._valid_epoch(epoch)

            # decompose result into the different losses
            total_train_loss = 0
            total_val_loss = 0
            self.logger.info('Epoch: {}/{}:'.format(epoch, self.epochs + 1))
            for (key_train, value_train), (key_val, value_val) in zip(result_train.items(), result_val.items()):

                total_train_loss += value_train
                total_val_loss   += value_val
                self.logger.info('--- {}: {}, {}: {}'.format(key_train, value_train, key_val, value_val))
                self.writer.add_scalars('Loss' + key_train.split('_')[0], {'train': value_train, 'val': value_val}, epoch)


            wandb.log({'epoch':epoch,
                       'lr': self.optimizer.param_groups[0]['lr'],
                       'total_loss': total_train_loss,
                       'total_val_loss': total_val_loss,
                       **result_train,
                       **result_val
                       })

            self.logger.info('------ Total loss: {}, Total validation loss: {}'.format(total_train_loss, total_val_loss))
            self.writer.add_scalars('Loss', {'train': total_train_loss, 'val': total_val_loss}, epoch)

            if result_val['lanes_val_loss'] < self.best_lane_value:
                self.save_lane = True
                self.best_lane_value = result_val['lanes_val_loss']
            else:
                self.save_lane = False

            if result_val['obj_val_loss'] < self.best_obj_value:
                self.save_obj = True
                self.best_obj_value = result_val['obj_val_loss']
            else:
                self.save_obj = False

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if total_val_loss < self.best_val:
                self.logger.info('*** New optimal model with validation performance increase of {} ***'.format(self.best_val-total_val_loss))
                self.best_val = total_val_loss
                self.save_best = True
                not_improved_count = 0

            else:
                self.logger.info('No performance increase over the last {} epochs'.format(not_improved_count))
                self.save_best = False
                not_improved_count += 1

            if total_train_loss < self.best_train:
                self.save_train = True
                self.best_train = total_train_loss
            else:
                self.save_train = False


            # if epoch % self.save_period == 0:
            self._save_checkpoint(epoch)

            if not_improved_count > self.early_stop:
                self.logger.info('Model performance has not improved over the last {} epochs. '.format(not_improved_count) +
                                 '\nTraining stops.')
                break

        self.writer.close()

    def _update_optimizer(self, loss):

        """
        Input: loss in batch
        Optimizer takes step in the direction of the loss
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_scheduler(self):

        """
        Scheduler takes one step
        """
        self.scheduler.step()

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        epoch: current epoch number
        best model with respect to training error, validation error, object detection error, and lane detection error is saved
        """

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }
        if epoch % self.save_period == 0:
            filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info('--------- Saved current epoch model.')

        if self.save_best:
            best_path = str(self.checkpoint_dir + '/model_best.pth')
            torch.save(state, best_path)
            self.logger.info('--------- *** Saved best model.')
            # wandb.save('best_model.h5')

        if self.save_obj:
            best_path = str(self.checkpoint_dir + '/best_obj_model.pth')
            torch.save(state, best_path)
            self.logger.info('--------- *** Saved best model wrt objects.')
            # wandb.save('best_obj_model.h5')

        if self.save_lane:
            best_path = str(self.checkpoint_dir + '/best_lane_model.pth')
            torch.save(state, best_path)
            self.logger.info('--------- *** Saved best model wrt objects.')
            # wandb.save('best_lane_model.h5')

        if self.save_train:
            best_path = str(self.checkpoint_dir + '/best_train_model.pth')
            torch.save(state, best_path)
            self.logger.info('--------- *** Saved best model wrt training.')



    def _resume_checkpoint(self):

        """
        Resume from saved checkpoints, path given in the config file

        """
        ##### Logger
        resume_path = str(self.config['trainer']['path_to_weights'])
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer and scheduler state from checkpoint only when optimizer type is not changed.
        if not self.config['datasets']['obj_and_lanes']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))


