import torch

from dataset.utils.obj_transformations import _tranpose_and_gather_feature
from trainer.losses import kitti_loss, lane_loss
from base import BaseTrainer
import wandb, cv2

torch.backends.cudnn.benchmark = True

SEED = 123
torch.manual_seed(SEED)

class Trainer(BaseTrainer):
    """
    Trainer class for training on one dataset that contains both lane annotations and object annotations
    """
    def __init__(self,
                 model,
                 train_data_loader, val_data_loader,
                 optimizer, scheduler,
                 config):

        super().__init__(model,
                         train_data_loader, val_data_loader,
                         optimizer, scheduler,
                         config)

        self.boundary_loss = lane_loss.get_loss_dict(use_aux=True,
                                           sim_loss_w=config['loss']['sim_loss_w'],
                                           shp_loss_w=config['loss']['shp_loss_w'])
        self.obj_loss_reg = kitti_loss.reg_loss
        self.obj_loss_neg = kitti_loss.neg_loss

        # Lists to keep track of batch losses
        self.avg_obj_loss = []
        self.avg_lanes_loss = []

        # Lists to keep track of validation batch losses
        self.avg_obj_val_loss = []
        self.avg_lanes_val_loss = []


        # Variable to update best loss
        self.best_loss = 10


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        iter_per_epoch = len(self.train_data_loader)
        for iter_num, batch in enumerate(self.train_data_loader):

            global_step = (epoch - 1) * iter_per_epoch + (iter_num+1)*self.config['finetuning_dataloader']['args']['batch_size']

            # if iter_num % 100 == 0:
            #     test = batch['test_img'][0].numpy()
            #     print(test.shape)
            #     wandb.log({"examples": [wandb.Image(cv2.cvtColor(test, cv2.COLOR_RGB2BGR))]})

            for k in batch:
                if k != 'meta' and k!='test_img':
                    batch[k] = batch[k].to('cuda', non_blocking=True)

            outputs_obj, out_lanes = self.model(batch['image'])

            cls_out, seg_out = out_lanes

            output_lanes = {'cls_out': cls_out, 'cls_label': batch['cls_label'], 'seg_out': seg_out,
                            'seg_label': batch['seg_label']}

            loss = 0
            for i in range(len(self.boundary_loss['name'])):
                data_src = self.boundary_loss['data_src'][i]

                datas = [output_lanes[src] for src in data_src]
                loss_cur = self.boundary_loss['op'][i](*datas)

                loss_contr = loss_cur * self.boundary_loss['weight'][i]
                loss += loss_contr

            lane_loss = loss
            self.avg_lanes_loss.append(lane_loss.item())

            hmap, regs, w_h_ = zip(*outputs_obj)
            regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
            w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

            hmap_loss = self.obj_loss_neg(hmap, batch['hmap'])
            reg_loss = self.obj_loss_reg(regs, batch['regs'], batch['ind_masks'])
            w_h_loss = self.obj_loss_reg(w_h_, batch['w_h_'], batch['ind_masks'])

            obj_loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss
            self.avg_obj_loss.append(obj_loss.item())


            wandb.log({'step': global_step,
                       'lr_step': self.optimizer.param_groups[0]['lr'],
                       'obj_step_loss': obj_loss.item(),
                       'lane_step_loss': lane_loss.item(),
                       'step_total_loss': obj_loss.item() + lane_loss.item()
                       })

            # Update optimizer ------------------------------------------------------
            self._update_optimizer(self.config['loss']['loss_multiplier'] *
                                   (obj_loss + self.config['loss']['loss_multiplier_lanes'] * lane_loss))

        # Update scheduler ----------------------------------------------------------
        self._update_scheduler()

        # Compute epoch average training loss ---------------------------------------

        obj_det_loss = sum(self.avg_obj_loss)/len(self.avg_obj_loss)
        lane_det_loss = sum(self.avg_lanes_loss)/len(self.avg_lanes_loss)

        self.avg_obj_loss = []
        self.avg_lanes_loss = []

        return {'obj_loss': obj_det_loss,
                'lanes_loss': lane_det_loss,
                }


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():

            for batch in self.val_data_loader:

                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to('cuda', non_blocking=True)

                outputs_obj, out_lanes = self.model(batch['image'])

                hmap, regs, w_h_ = zip(*outputs_obj)
                regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

                hmap_loss = self.obj_loss_neg(hmap, batch['hmap'])
                reg_loss = self.obj_loss_reg(regs, batch['regs'], batch['ind_masks'])
                w_h_loss = self.obj_loss_reg(w_h_, batch['w_h_'], batch['ind_masks'])

                obj_val_loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss
                self.avg_obj_val_loss.append(obj_val_loss.item())

                cls_out, seg_out = out_lanes

                output_lanes = {'cls_out': cls_out, 'cls_label': batch['cls_label'],
                                'seg_out': seg_out, 'seg_label': batch['seg_label']}

                loss = 0
                for i in range(len(self.boundary_loss['name'])):
                    data_src = self.boundary_loss['data_src'][i]

                    datas = [output_lanes[src] for src in data_src]

                    loss_cur = self.boundary_loss['op'][i](*datas)

                    loss_contr = loss_cur * self.boundary_loss['weight'][i]
                    loss += loss_contr

                lane_val_loss = loss
                self.avg_lanes_val_loss.append(lane_val_loss.item())

        obj_det_val_loss = sum(self.avg_obj_val_loss)/len(self.avg_obj_val_loss)
        lane_det_val_loss = sum(self.avg_lanes_val_loss)/len(self.avg_lanes_val_loss)

        ### Write validation loss to logger

        self.avg_obj_val_loss = []
        self.avg_lanes_val_loss = []

        return {'obj_val_loss': obj_det_val_loss,
                'lanes_val_loss': lane_det_val_loss}

    def _progress(self, batch_idx, total_batches):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = total_batches
        return base.format(current, total, 100.0 * current / total)
