from base import BaseDataLoader
from dataset.object.obj import ObjectDataset
from dataset.lanes.laneOrig import LaneDataset
from dataset.objectsAndLanes.dataset import JointDataset

class ObjectDataLoader(BaseDataLoader):

    def __init__(self, config, mode):
        """
        config: config file with hyperparameters
        mode: weather the data is used for training or validating
        """
        dataset = ObjectDataset(config, mode)
        specificDataset = config['datasets']['obj'] + '_dataloader'
        super().__init__(config[specificDataset]['args'], dataset)

class LanesDataLoader(BaseDataLoader):
    def __init__(self, config, mode):
        """
        config: config file with hyperparameters
        mode: weather the data is used for training or validating
        """
        dataset = LaneDataset(config, mode)
        specificDataset = config['datasets']['lanes'] + '_dataloader'
        super().__init__(config[specificDataset]['args'], dataset)

class LabeledDataLoader(BaseDataLoader):
    def __init__(self, config, mode):
        """
        config: config file with hyperparameters
        mode: weather the data is used for training or validating
        """
        dataset = JointDataset(config, mode)
        super().__init__(config['finetuning_dataloader']['args'], dataset)





