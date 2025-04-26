import argparse
import open_clip
import torch
import logging
import time

import os
import yaml

import numpy as np
import pandas as pd
import torch
import ruamel.yaml as yaml

from train.two_step import first_step, second_step
from train.common import train, train_eval

from trainer.ClipWrapper import ClipObject
from trainer.classifiers import NNClassifier, PrototypeClassifier, CLIPClassifier, build_CLIPclassifier, build_prototype_classifier
from Dataset.FungiTastic import FungiTastic

def setup_logging(log_path, debug):
    logger = logging.getLogger()
    if not debug:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_path, 'log')
    else:
        logger.setLevel(logging.DEBUG)
        curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        log_path = os.path.join(log_path, f'debug_{curr_time}')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Log to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Log to file
    if log_path:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            raise ValueError(f'Log path {log_path} already exists. ')
        log_file = os.path.join(log_path, 'log.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return log_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument(
        '--device', 
        type=str,
        default='cuda:1',
        help="Specify the CUDA device number, e.g., 'cuda:0', 'cuda:1', etc."
    )
    parser.add_argument(
        '--c',
        type=str,
        default='config/CLIPclassifer.yaml',
        help='Path to the configuration file'
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default='/research/nfs_chao_209/fungi-clef-2025',
    )
    parser.add_argument(
        "--classifer_ckpt_dir",
        type=str,
        default='/research/nfs_chao_209/snapshot/MODEL/visual_LoRA',
        help="Path to the classifier checkpoint directory",
    )
    parser.add_argument(
        "--classifer_ckpt_path",
        type=str,
        # default='/research/nfs_chao_209/snapshot/MODEL/visual_LoRA/fungi-clef25_bioclip_two-step_lora_64_classifer/fungi-clef25_bioclip_two-step_lora_32_classifer/epoch_80.pth',
        help="Path to the classifier checkpoint path",
    )
    args = parser.parse_args()
    if args.c:
        with open(args.c, 'r') as f:
            yml = yaml.YAML(typ='rt')
            config = yml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args

if __name__ == "__main__":
    args = parse_arguments()
    device = args.device
    model_cfg = args.model_cfg
    train_cfg = args.train_cfg
    optimizer_cfg = train_cfg['optimizer_cfg']
    classifer_ckpt_dir = args.classifer_ckpt_dir

    clip_model = ClipObject(model_cfg, device)
    if model_cfg['fine_tune'] == 'LoRA':
        clip_model.unfreeze_param_LoRA()
    else:
        clip_model.unfreeze_param_fft()

    optimizer = torch.optim.AdamW(
        clip_model.model.parameters(),
        lr=1e-4,
        weight_decay=optimizer_cfg['wd'],
        betas=(optimizer_cfg['clip_beta1'], optimizer_cfg['clip_beta2']),
    )

    train_dataset = FungiTastic(args.data_root, split='train', usage='training')
    val_dataset = FungiTastic(args.data_root, split='val', usage='training')

    # print("Printing unfrozen parameters: ")
    # unfreeze_params = [(n, p) for n, p in clip_model.named_parameters() if p.requires_grad]
    # for name, param in unfreeze_params:
    #     num_elements = param.numel()
    #     print(f"Parameter name: {name}, Shape: {param.shape}, Number of elements: {num_elements}")

    train(clip_model, device, optimizer, model_cfg, train_cfg, train_dataset, val_dataset, classifer_ckpt_dir)
    train_eval(clip_model, device, optimizer, model_cfg, train_cfg, val_dataset, classifer_ckpt_dir)

    # train LoRA
    # first_step(clip_model, device, optimizer, model_cfg, train_cfg, train_dataset)

    # class_set = train_dataset.class_set
    # classifier = build_prototype_classifier(clip_model, train_dataset)
    # if args.classifer_ckpt_path:
    #     classifier.load(args.classifer_ckpt_path)
    # classifier.unfreeze_head()

    # optimizer = torch.optim.AdamW(
    #         classifier.parameters(),
    #         lr=1e-4,
    #         weight_decay=optimizer_cfg['wd'],
    #         betas=(optimizer_cfg['clip_beta1'], optimizer_cfg['clip_beta2']),
    #     )

    # print("Printing unfrozen parameters: ")
    # unfreeze_params = [(n, p) for n, p in classifier.named_parameters() if p.requires_grad]
    # for name, param in unfreeze_params:
    #     num_elements = param.numel()
    #     print(f"Parameter name: {name}, Shape: {param.shape}, Number of elements: {num_elements}")

    # # train classifier head
    # second_step(classifier, device, optimizer, model_cfg, train_cfg, train_dataset, val_dataset, classifer_ckpt_dir)
