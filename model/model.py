import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.backbone import resnet
from model.heads import AegisLaneHead, AegisObjHead


class AegisMTModel(BaseModel):
    def __init__(self, bbone, head_lane, head_obj):
        """
        bbone: backbone architecture
        head_lane: lane detection architecture
        head_obj: object detection architecture
        returns hard parameter sharing structure:

                       ---- object-detection
        backbone ------
                       ---- lane-detection
        """
        super().__init__(backbone=bbone, lane_heads=head_lane, obj_head=head_obj)

        self.lanes_head = self.heads['lane_heads']
        self.obj_head = self.heads['obj_head']


    def forward(self, x):

        x2, x3, features = self.backbone(x)

        out_lanes = self.lanes_head(x2, x3, features)

        out_obj = self.obj_head(features)

        return out_obj, out_lanes
