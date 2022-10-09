import torch

from dataset.utils.obj_transformations import _tranpose_and_gather_feature
from trainer.losses import kitti_loss, lane_loss
from base import BaseTrainer
import wandb

torch.backends.cudnn.benchmark = True

SEED = 123
torch.manual_seed(SEED)

class Trainer2Datasets(BaseTrainer):
    """
    Trainer class. Similar to Trainer2DStochastic, but without stochastic component: batch of lane-annotated images
    and batch of object-annotaated image is alternatively used
    """
    def __init__(self,
                 model,
                 lanes_data_loader, lanes_val_data_loader,
                 kitti_data_loader, kitti_val_data_loader,
                 optimizer, scheduler,
                 config):

        super().__init__(model,
                         lanes_data_loader, lanes_val_data_loader,
                         optimizer, scheduler,
                         config)

        self.lanes_data_loader = lanes_data_loader
        self.lanes_val_loader = lanes_val_data_loader
        self.kitti_data_loader = kitti_data_loader
        self.kitti_val_loader = kitti_val_data_loader


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

        # Variables to save losses of each bath
        self.kitti_loss = None
        self.lane_loss = None
        self.kitti_val_loss = None
        self.lane_val_loss = None

        # Variables to keep average training losses
        self.obj_det_loss = 0
        self.lane_det_loss = 0
        self.obj_det_val_loss = 0
        self.lane_det_val_loss = 0
        #self.total_loss = 0

        # Variable to update best loss
        self.best_loss = 10


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        kitti_data_iter = iter(self.kitti_data_loader)
        lane_data_iter = iter(self.lanes_data_loader)
        iter_per_epoch = min(len(self.kitti_data_loader), len(self.lanes_data_loader), 5000)

        for iter_num in range(1, iter_per_epoch + 1):

            global_step = (epoch - 1) * iter_per_epoch + (iter_num-1)*self.config['kitti_dataloader']['args']['batch_size']

            # Train on objects  ------------------------------------------------------
            try:
                batch_obj = next(kitti_data_iter)

            except StopIteration:
                kitti_data_iter = iter(self.kitti_data_loader)
                batch_obj = next(kitti_data_iter)

            self.kitti_loss = self.config['loss']['loss_multiplier']*self._predict_obj_compute_loss(batch_obj)

            self.avg_obj_loss.append(self.kitti_loss.item())

            # Train on lanes --------------------------------------------------------
            try:
                batch_lane = next(lane_data_iter)
            except StopIteration:
                lane_data_iter = iter(self.lanes_data_loader)
                batch_lane = next(lane_data_iter)


            self.lane_loss = self.config['loss']['loss_multiplier']*self._predict_lane_compute_loss(batch_lane)
            self.avg_lanes_loss.append(self.lane_loss.item())

            wandb.log({'step': global_step,
                       'lr_step': self.optimizer.param_groups[0]['lr'],
                       'obj_step_loss': self.kitti_loss.item(),
                       'lane_step_loss': self.lane_loss.item(),
                       'step_total_loss': self.lane_loss.item() + self.kitti_loss.item()
                       })

            # Update optimizer ------------------------------------------------------
            self._update_optimizer(self.kitti_loss + self.config['loss']['loss_multiplier_lanes']*self.lane_loss)

            # print('Epoch {}, batch {}/{}'.format(epoch, iter_num, iter_per_epoch))

        # Update scheduler ----------------------------------------------------------
        self._update_scheduler()

        # Compute epoch average training loss ---------------------------------------
        self.obj_det_loss = sum(self.avg_obj_loss)/len(self.avg_obj_loss)
        self.lane_det_loss = sum(self.avg_lanes_loss)/len(self.avg_lanes_loss)

        ### Write training loss to logger

        self.avg_obj_loss = []
        self.avg_lanes_loss = []

        return {'obj_loss': self.obj_det_loss,
                'lanes_loss': self.lane_det_loss,
                }


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self._validate_kitti()
        self._validate_lanes()

        self.obj_det_val_loss = sum(self.avg_obj_val_loss)/len(self.avg_obj_val_loss)
        self.lane_det_val_loss = sum(self.avg_lanes_val_loss)/len(self.avg_lanes_val_loss)

        ### Write validation loss to logger

        self.avg_obj_val_loss = []
        self.avg_lanes_val_loss = []

        return {'obj_val_loss': self.obj_det_val_loss,
                'lanes_val_loss': self.lane_det_val_loss}


    def _validate_kitti(self):
        """
        Validate object detection after training an epoch
        :param epoch: Integer, current training epoch.
        """
        with torch.no_grad():

            for batch in self.kitti_val_loader:

                self.kitti_val_loss = self._predict_obj_compute_loss(batch)

                self.avg_obj_val_loss.append(self.kitti_val_loss.item())

    def _validate_lanes(self):
        """
        Validate lane detection after training an epoch
        :param epoch: Integer, current training epoch.
        """
        with torch.no_grad():
            for batch in self.lanes_val_loader:

                self.lane_val_loss = self._predict_lane_compute_loss(batch)

                self.avg_lanes_val_loss.append(self.lane_val_loss.item())


    def _predict_obj_compute_loss(self, batch):
        """
        Predicts objects in batch
        :param batch: batch of images
        """
        for k in batch:
            if k != 'meta' and k!='img_id':
                batch[k] = batch[k].to('cuda', non_blocking=True)

        outputs_obj, outputs_lanes = self.model(batch['image'])
        hmap, regs, w_h_ = zip(*outputs_obj)
        regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
        w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

        hmap_loss = self.obj_loss_neg(hmap, batch['hmap'])
        reg_loss = self.obj_loss_reg(regs, batch['regs'], batch['ind_masks'])
        w_h_loss = self.obj_loss_reg(w_h_, batch['w_h_'], batch['ind_masks'])

        return hmap_loss + 1 * reg_loss + 0.1 * w_h_loss


    def _predict_lane_compute_loss(self, batch):
        """
        Predicts lanes in batch
        :param batch: batch of images
        """
        img, cls_label, seg_label = batch
        img, cls_label, seg_label = img.cuda(), cls_label.cuda(), seg_label.cuda()
        cls_out, seg_out = self.model(img)[1]

        output_lanes = {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}

        loss = 0
        for i in range(len(self.boundary_loss['name'])):

            # name = self.boundary_loss['name'][i]
            data_src = self.boundary_loss['data_src'][i]

            datas = [output_lanes[src] for src in data_src]

            loss_cur = self.boundary_loss['op'][i](*datas)

            loss_contr = loss_cur * self.boundary_loss['weight'][i]
            loss += loss_contr

            # wandb.log({'global_step': step,
            #            name: loss_contr.item(),
            #            })

        return loss



    def _progress(self, batch_idx, total_batches):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = total_batches
        return base.format(current, total, 100.0 * current / total)
