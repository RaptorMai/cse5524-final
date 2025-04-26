import sys
import argparse
import open_clip
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

import os
import json
import ruamel.yaml as yaml
from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np
import pandas as pd
import torch
import faiss

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as tfms
import torchvision.transforms as T

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union
import importlib

from trainer.classifiers import NNClassifier, PrototypeClassifier, CLIPClassifier

from trainer.ClipWrapper import ClipObject
from Dataset.FungiTastic import FungiTastic

def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument(
        '--device', 
        type=str,
        default='cuda:0',
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
        "--classifier_ckpt",
        type=str,
        help="Path to the classifier checkpoint",
        default='/research/nfs_chao_209/snapshot/MODEL/visual_LoRA/fungi-clef25_bioclip_two-step_lora_64_classifer/fungi-clef25_bioclip_two-step_lora_32_classifer/epoch_100.pth',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='/home/tian.855/fungi/result/lora32-5-NN.csv',
        help="Path to save the output",
    )
    args = parser.parse_args()
    if args.c:
        with open(args.c, 'r') as f:
            yml = yaml.YAML(typ='rt')
            config = yml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args

def generate_embeddings(dataset, tp, model):

    idxs = np.arange(len(dataset))
    im_names, embs = [], []
    for idx in tqdm(idxs):
        img, text, label, file_path = dataset[idx]

        with torch.no_grad():
            feat = model.extract_img_features(img)

        im_names.append(os.path.basename(file_path))
        embs.append(feat.detach().cpu().numpy())

    embeddings = pd.DataFrame({'filename': im_names, 'embedding': embs})

    return embeddings

if __name__ == "__main__":
    args = parse_arguments()
    ### Load the datasets
    data_path = args.data_root
    model_cfg = args.model_cfg
    device = args.device

    output_path = Path(args.output_path) if args.output_path else os.path.join(data_path, "submission")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = FungiTastic(root=data_path, split='train', usage='testing', transform=None)
    test_dataset = FungiTastic(root=data_path, split='test', usage='testing',transform=None)
    
    print("Loading BioCLIP model...")
    bioclip = ClipObject(model_cfg, device)
    print("BioCLIP model loaded successfully")
    bioclip.model.eval()
    print("Loading classifier...")
    classifier = CLIPClassifier(bioclip)
    classifier.load(args.classifier_ckpt)
    print("Classifier loaded successfully")

    test_embeddings = generate_embeddings(test_dataset, 'test', classifier)
    test_dataset.add_embeddings(test_embeddings)

    train_embeddings = generate_embeddings(train_dataset, 'train', classifier)
    train_dataset.add_embeddings(train_embeddings)

    #### 5-NN
    classifier = NNClassifier(train_dataset, device='cuda:0')

    cls, conf = classifier.make_prediction(np.array(test_dataset.df.embedding.values.tolist(), dtype=np.float32).squeeze())
    
    # Convert 2D array of predictions to a list of strings (space-separated class IDs)
    if cls.ndim > 1:
        # Convert each row of the 2D array to a space-separated string
        preds_list = [' '.join(map(str, row)) for row in cls]
    else:
        # If it's a 1D array, just convert each item to string
        preds_list = [str(c) for c in cls]
    
    # Assign the processed predictions to the DataFrame
    test_dataset.df["preds"] = preds_list
    
    # Group by observationID and gather unique predictions for each observation
    # But limit to 5 predictions per observationID
    submission_data = []
    
    for obs_id, group in test_dataset.df.groupby("observationID"):
        # Combine all predictions
        all_preds = " ".join(group["preds"])
        # Split into individual prediction IDs
        pred_ids = all_preds.split()
        
        # Count frequency of each prediction
        from collections import Counter
        pred_counts = Counter(pred_ids)
        
        # Get the 5 most common predictions
        top_preds = [pred for pred, _ in pred_counts.most_common(5)]
        
        # Format as space-separated string
        pred_str = " ".join(top_preds)
        
        submission_data.append({"observationId": obs_id, "predictions": pred_str})
    
    submission = pd.DataFrame(submission_data)
    submission.to_csv(args.output_path, index=None)
