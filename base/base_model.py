import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, **kwargs):

        """
        Inputs: backbone and heads of Multi-Task Networks, all given as key:value pairs
        """
        super(BaseModel, self).__init__()
        self.backbone = kwargs['backbone']

        self.heads = {key: value for key,value in kwargs.items() if key != 'config' and key != 'backbone'}


    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
