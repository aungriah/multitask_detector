import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json
import json


class ConfigParser:
    def __init__(self, config, resume=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        self._config = config
        exper_name = self.config['name']
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        # set save_dir where trained model and log will be saved.
        save_dir = os.path.join(self._config['trainer']['save_dir'], exper_name, run_id)
        log_dir = os.path.join(save_dir, 'log_dir')

        self._save_dir = save_dir
        self._log_dir = log_dir
        # make directory for saving checkpoints
        checkpoint_dir = os.path.join(save_dir, 'checkpoint_dir')
        inference_dir = os.path.join(save_dir, 'inference_dir')

        modification = {'run': config['tasks'] + '_' + run_id,
                        'code': os.path.join(save_dir, 'code'),
                        'trainer;log_dir': log_dir,
                        'trainer;checkpoint_dir': checkpoint_dir,
                        'trainer;inference_dir': inference_dir}

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(inference_dir, exist_ok=True)

        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume


        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir + '/config.json')

        with open(self.save_dir + '/config.json') as cfg:
            self.config_dict = json.load(cfg)



        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """

        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.mode == 'test':
            cfg_fname = Path(args.config)
            config = read_json(cfg_fname)

            config['arch']['use_aux'] = False
            config['trainer']['path_to_weights'] = args.weights
            config['image_dir'] = args.imgdir

            os.makedirs(args.save, exist_ok=True)
            config['save_dir'] = args.save
            config['device'] = args.device

            return cls(config)

        if args.mode == 'speed_test':
            cfg_fname = Path(args.config)
            config = read_json(cfg_fname)

            config['arch']['use_aux'] = False
            config['arch']['backbone_layers'] = str(args.backbone)
            config['arch']['input_width'] = int(args.width)
            config['arch']['input_height'] = int(args.height)

            return cls(config)


        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'

        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        if args.mode == 'train':
            config['arch']['use_aux'] = True

        # parse custom cli options into dictionary
        return cls(config, resume)





    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
