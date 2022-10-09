import numpy as np
import cv2
import random

import torch
import torch.nn.functional as F

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)

def _gather_feature(feat, ind, mask=None):
  """
  feat: feature detected by network
  ind: indices of batches, indicating location of an objct
  returns error associated to that feature
  """
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
    mask = mask.unsqueeze(2).expand_as(feat)
    feat = feat[mask]
    feat = feat.view(-1, dim)
  return feat


def _tranpose_and_gather_feature(feat, ind):
  """
  feat: feature detected by network
  ind: indices of batches, indicating location of an objct
  returns error associated to that featuree
  """
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feature(feat, ind)
  return feat

def _nms(heat, kernel=3):
  """
  hmap: heat-map predicted by network
  Perform non-maximum supression on heat-map
  """
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  return heat * keep


def _topk(scores, K=40):
  """
  scores: certainty of each prediction made by network
  K: threshold
  returns K best outputs of the network based on the scores
  """
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(hmap, regs, w_h_, K=100):
  """
  hmap: heatmap predicted by network
  regs: regularization error predicted by network
  w_h_L dimension inferred by network
  return scoree, class, and bounding box of each prediction
  """
  batch, cat, height, width = hmap.shape
  hmap=torch.sigmoid(hmap)

  # if flip test
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
    w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
    regs = regs[0:1]

  batch = 1

  hmap = _nms(hmap)  # perform nms on heatmaps

  scores, inds, clses, ys, xs = _topk(hmap, K=K)

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  w_h_ = _tranpose_and_gather_feature(w_h_, inds)
  w_h_ = w_h_.view(batch, K, 2)

  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)
  bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2)
  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections

def flip_tensor(x):
  """
  x: tensor
  returns flipped tensor
  """
  return torch.flip(x, [3])

def flip(img):
    """
    img: image
    returns flipped image
    """
    return img[:, :, ::-1].copy()

def gaussian_radius(det_size, min_overlap=0.7):
    """
    det_size: siz of bounding box of ground-truth object
    min_overlap: parameter to control radius
    returns radius to draw gaussian aroung center of ground-truth object
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    # r1 = (b1 + sq1) / 2 #
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    # r2 = (b2 + sq2) / 2
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    # r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    shape: diameter
    returns 2D  gaussain
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    heatmap: ground-truth heatmap to draw gaussian on
    center: center of the ground-truth object
    radius: radius of peak to be drawn
    k: not needed
    returns heatmap with a peak of radius (radius) at centere (center)
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
#     diameter = 2 * radius + 1
#     gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
#     value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
#     dim = value.shape[0]
#     reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
#     if is_offset and dim == 2:
#         delta = np.arange(diameter * 2 + 1) - radius
#         reg[0] = reg[0] - delta.reshape(1, -1)
#         reg[1] = reg[1] - delta.reshape(-1, 1)
#
#     x, y = int(center[0]), int(center[1])
#
#     height, width = heatmap.shape[0:2]
#
#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)
#
#     masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
#     masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom,
#                       radius - left:radius + right]
#     masked_reg = reg[:, radius - top:radius + bottom,
#                  radius - left:radius + right]
#     if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
#         idx = (masked_gaussian >= masked_heatmap).reshape(
#             1, masked_gaussian.shape[0], masked_gaussian.shape[1])
#         masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
#     regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
#     return regmap
#
#
# def draw_msra_gaussian(heatmap, center, sigma):
#     tmp_size = sigma * 3
#     mu_x = int(center[0] + 0.5)
#     mu_y = int(center[1] + 0.5)
#     w, h = heatmap.shape[0], heatmap.shape[1]
#     ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
#     br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
#     if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
#         return heatmap
#     size = 2 * tmp_size + 1
#     x = np.arange(0, size, 1, np.float32)
#     y = x[:, np.newaxis]
#     x0 = y0 = size // 2
#     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#     g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
#     g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
#     img_x = max(0, ul[0]), min(br[0], h)
#     img_y = max(0, ul[1]), min(br[1], w)
#     heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
#         heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
#         g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
#     return heatmap

