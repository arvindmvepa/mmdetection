from __future__ import division

import argparse
import os
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch
import os
import json
import copy


os.environ["PYTHONUNBUFFERED"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def add_to_config(nested_k, v, config):
    k = nested_k[0]
    if k not in config:
        # copy over nested dictionary to config
        temp = dict()
        for i, k_ in enumerate(nested_k[::-1]):
            if i == 0:
                temp[k_] = v
                temp_ = dict()
            else:
                temp_[k_] = copy.deepcopy(temp)
                temp = copy.deepcopy(temp_)
                temp_ = dict()
        config.update(temp)
    else:
        config[k] = add_to_config(nested_k[1:], v, config[k])
    return config


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    try:
        with open('/opt/ml/input/config/hyperparameters.json') as json_file:
            load_config = json.load(json_file)
    except Exception as e:
        print(e)

    # add hyper-parameters from sagemaker
    for k,v in load_config.items():
        nested_k = k.split(".")
        cfg = add_to_config(nested_k, v, cfg)

    cfg.work_dir = '/opt/ml/model/'
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if cfg.seed is not None:
        logger.info('Set random seed to {}'.format(cfg.seed))
        set_random_seed(cfg.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = build_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=cfg.validate,
        logger=logger)


if __name__ == '__main__':
    main()
