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
# Add dimensionality reduction imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
        default='/home/tian.855/fungi/result/lora32-ProClass.csv',
        help="Path to save the output",
    )
    parser.add_argument(
        "--reduction_method",
        type=str,
        default='tsne',
        choices=['tsne', 'pca'],
        help="Dimensionality reduction method for visualization",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Path to save the embedding visualization plot",
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

def reduce_dimensionality(embeddings, method='tsne', random_state=42):
    """
    Reduce dimensionality of embeddings for visualization.
    
    Args:
        embeddings: Numpy array of embeddings
        method: 'tsne' or 'pca'
        random_state: Random seed for reproducibility
        
    Returns:
        2D embeddings
    """
    if method.lower() == 'tsne':
        print(f"Applying t-SNE dimensionality reduction...")
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method.lower() == 'pca':
        print(f"Applying PCA dimensionality reduction...")
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    return reducer.fit_transform(embeddings)

def plot_embeddings(reduced_embeddings, labels=None, title="Embedding Visualization", 
                   figsize=(12, 10), save_path=None, method='t-SNE'):
    """
    Plot reduced dimensionality embeddings.
    
    Args:
        reduced_embeddings: 2D numpy array of reduced embeddings
        labels: Optional labels for coloring points
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        method: Reduction method used (for title)
    """
    plt.figure(figsize=figsize)
    
    if labels is not None:
        # If we have labels, use them for coloring
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            indices = labels == label
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                color=colors[i],
                label=f"Class {label}",
                alpha=0.7,
                s=10
            )
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
    else:
        # If no labels, use a single color
        plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            alpha=0.7,
            s=10
        )
    
    plt.title(f"{title} ({method} projection)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

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

    train_embeddings = generate_embeddings(train_dataset, 'test', classifier)
    train_dataset.add_embeddings(train_embeddings)
    

    
    print("Visualization complete!")