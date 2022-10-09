import torch
import torch.nn as nn
import torch.nn.functional as F


class KittiLoss():
    def __init__(self):

        self.neg_loss = neg_loss
        self.reg_loss = reg_loss

def neg_loss_slow(preds, targets):
  """
  preds: network prediction
  targets: ground-truth objects
  returns focal loss
  """
  pos_inds = targets == 1  # todo targets > 1-epsilon ?
  neg_inds = targets < 1  # todo targets < 1-epsilon ?

  neg_weights = torch.pow(1 - targets[neg_inds], 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def neg_loss(preds, targets):
  '''
  Modified focal loss. Exactly the same as CornerNet.
  Runs faster and costs a little bit more memory
  Arguments:
  preds: predictions (B x c x h x w)
  gt_regr: grount-truth (B x c x h x w)
  returns loss between prediction and targets
  '''
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)


def reg_loss(regs, gt_regs, mask):
  """
  regs: predicted offset
  gt_regs: ground-truth offset
  mask: ground-truth mask with indices locating objects
  returns offset loss
  """
  mask = mask[:, :, None].expand_as(gt_regs).float()
  loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
  return loss / len(regs)