import os, json, pathspec
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

def cp_projects(config):
    """
    config: parameters used for training, including where to save copies of all files
    Copies all files in the run to the director included in the config file
    """
    to_path = config["location"]
    with open('./.gitignore','r') as fp:
        ign = fp.read()
    ign += '\n.git'
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
    all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files if '.py' or '.sh' or '.json' in name}
    matches = spec.match_files(all_files)
    matches = set(matches)
    to_cp_files = all_files - matches
    # to_cp_files = [f[2:] for f in to_cp_files]
    # pdb.set_trace()
    for f in to_cp_files:
        dirs = os.path.join(config['code'],os.path.split(f[2:])[0])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        os.system('cp %s %s'%(f,os.path.join(config['code'],f[2:])))

# Functions listed below are unnecessary

def nested_dict_to_dict(nested_dict, dictionary):
    for key, value in nested_dict.items():

        if isinstance(value, dict):
            partial = {}
            for sub_key, sub_value in value.items():
                partial[key +'-'+sub_key] = sub_value
            nested_dict_to_dict(partial, dictionary)
        else:
            dictionary[key]=value

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
