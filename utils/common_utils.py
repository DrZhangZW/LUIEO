
import random
import numpy as np
from utils import logger
import os
from torch import nn
from torch import Tensor
import glob,re
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
#================================Load yaml====================================
import yaml
import collections
def flatten_yaml_as_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_config_file(opts):
    config_file_name = opts.common_config_file
    if config_file_name is None:
        return opts
    with open(config_file_name, 'r') as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            flat_cfg = flatten_yaml_as_dict(cfg)
            logger.log("print parser setting")
            logger.log("parser-name   value")
            for k, v in flat_cfg.items():
                #if hasattr(opts, k):
                setattr(opts, k, v)
                print(f'{k}:{v}' )
        except yaml.YAMLError as exc:
            logger.warning('Error while loading config file: {}'.format(config_file_name))
            logger.warning('Error message: {}'.format(str(exc)))
    return opts
#-------------------------device-----------------------------------
#==============================================================================
import torch
def device_setup(opts):
    random_seed = getattr(opts, "common_seed", 0)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    is_master_node = True
    if is_master_node:
        logger.log('Random seeds are set to {}'.format(random_seed))
        logger.log('Using PyTorch version {}'.format(torch.__version__))

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        if is_master_node:
            logger.warning('No GPUs available. Using CPU')
        device = torch.device('cpu')
        n_gpus = 0
    else:
        if is_master_node:
            logger.log('Available GPUs: {}'.format(n_gpus))
        device = torch.device('cuda')

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            torch.backends.cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            if is_master_node:
                logger.log('CUDNN is enabled')

    setattr(opts, "dev_device", device)
    setattr(opts, "dev_num_gpus", n_gpus)
    return opts

def create_directories(dir_path: str, is_master_node: bool) -> None:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        if is_master_node:
            logger.log('Directory created at: {}'.format(dir_path))
    else:
        if is_master_node:
            logger.log('Directory exists at: {}'.format(dir_path))
def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path
#======================================================================================
import re
import pkg_resources as pkg
def check_version(current: str = '0.0.0',
                  required: str = '0.0.0',
                  name: str = 'version ',
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        # check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
    """
    current = pkg.parse_version(current)
    constraints = re.findall(r'([<>!=]{1,2}\s*\d+\.\d+)', required) or [f'>={required}']

    result = True
    for constraint in constraints:
        op, version = re.match(r'([<>!=]{1,2})\s*(\d+\.\d+)', constraint).groups()
        version = pkg.parse_version(version)
        if op == '==' and current != version:
            result = False
        elif op == '!=' and current == version:
            result = False
        elif op == '>=' and not (current >= version):
            result = False
        elif op == '<=' and not (current <= version):
            result = False
        elif op == '>' and not (current > version):
            result = False
        elif op == '<' and not (current < version):
            result = False
    return result
