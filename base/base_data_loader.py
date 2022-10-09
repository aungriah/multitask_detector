import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset_cfg, dataset):
        self.dataset = dataset
        self.data_dir = dataset_cfg['data_dir']
        self.batch_size = dataset_cfg['batch_size']
        self.shuffle = dataset_cfg['shuffle']
        self.validation_split = dataset_cfg['validation_split']
        self.num_workers = dataset_cfg['num_workers']
        self.drop_last = dataset_cfg['drop_last']

        super().__init__(dataset=self.dataset,
                         batch_size=self.batch_size,
                         shuffle= self.shuffle,
                         num_workers=self.num_workers,
                         pin_memory=True,
                         drop_last=self.drop_last
                         )
        # super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)