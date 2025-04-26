import os
import json
import yaml
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
import open_clip

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union
import importlib

class FungiTastic(torch.nn.Module):
    """
    Dataset class for the FewShot subset of the Danish Fungi dataset (size 300, closed-set).

    This dataset loader supports training, validation, and testing splits, and provides
    convenient access to images, class IDs, and file paths. It also supports optional
    image transformations.
    """

    SPLIT2STR = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
    IMG_DIRS = ['300p', '500p', '720p', 'fullsize']

    def __init__(self, root: str, split: str = 'val', usage='testing', transform=None, tokenizer=None):
        """
        Initializes the FungiTastic dataset.

        Args:
            root (str): The root directory of the dataset.
            split (str, optional): The dataset split to use. Must be one of {'train', 'val', 'test'}.
                Defaults to 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.df = self._get_df(usage, root, split)
        self.usage = usage

        if self.split != 'test':
            self.get_prompts()

        assert "image_path" in self.df
        if self.split != 'test':
            assert "category_id" in self.df
            self.n_classes = len(self.df['category_id'].unique())
            self.category_id2label = {
                k: v[0] for k, v in self.df.groupby('category_id')['scientificName'].unique().to_dict().items()
            }
            self.label2category_id = {
                v: k for k, v in self.category_id2label.items()
            }

    def get_prompts(self):
        """
        Returns the class names for the dataset.
        """
        prompt_set = []
        scientific_names = self.df['scientificName'].tolist()
        common_names = self.df['species'].tolist()
        for i, sci_name in enumerate(scientific_names):
            prompt_set.append(f"A picture of {sci_name} with common name {common_names[i]}")
        
        self.prompt_set = prompt_set
        ## Class set
        self.class_set = list(set(self.prompt_set))

    def add_embeddings(self, embeddings: pd.DataFrame):
        """
        Updates the dataset instance with new embeddings.

        Args:
            embeddings (pd.DataFrame): A DataFrame containing an 'embedding' column.
                                       It must align with `self.df` in terms of indexing.
        """
        assert isinstance(embeddings, pd.DataFrame), "Embeddings must be a pandas DataFrame."
        assert "embedding" in embeddings.columns, "Embeddings DataFrame must have an 'embedding' column."
        assert len(embeddings) == len(self.df), "Embeddings must match dataset length."

        self.df = pd.merge(self.df, embeddings, on="filename", how="inner")

    def get_embeddings_for_class(self, id):
        # return the embeddings for class class_idx
        class_idxs = self.df[self.df['category_id'] == id].index
        return self.df.iloc[class_idxs]['embedding']
    
    @staticmethod
    def _get_df(usage: str, data_path: str, split: str) -> pd.DataFrame:
        """
        Loads the dataset metadata as a pandas DataFrame.

        Args:
            data_path (str): The root directory where the dataset is stored.
            split (str): The dataset split to load. Must be one of {'train', 'val', 'test'}.

        Returns:
            pd.DataFrame: A DataFrame containing metadata and file paths for the split.
        """
        df_path = os.path.join(
            data_path,
            "metadata",
            "FungiTastic-FewShot",
            f"FungiTastic-FewShot-{FungiTastic.SPLIT2STR[split]}.csv"
        )
        df = pd.read_csv(df_path)
        df["image_path"] = df.filename.apply(
            lambda x: os.path.join(data_path, "FungiTastic-FewShot", split, '300p', x)
        )
        
        return df

    def __getitem__(self, idx: int):
        """
        Retrieves a single data sample by index.
    
        Args:
            idx (int): Index of the sample to retrieve.
            ret_image (bool, optional): Whether to explicitly return the image. Defaults to False.
    
        Returns:
            tuple:
                - If embeddings exist: (image?, embedding, category_id, file_path)
                - If no embeddings: (image, category_id, file_path) (original version)
        """
        file_path = self.df["image_path"].iloc[idx].replace('FungiTastic-FewShot', 'images/FungiTastic-FewShot')
    
        if self.split != 'test':
            category_id = self.df["category_id"].iloc[idx]
        else:
            category_id = None

        if self.usage != 'training':
            image = Image.open(file_path)
        else:
            image = file_path
        
        if self.split != 'test':
            text = self.prompt_set[idx]
        else:
            text = None

        if self.transform:
            image = self.transform(image)
        
        if self.tokenizer:
            text = self.tokenizer(self.prompt_set[idx])

        # Check if embeddings exist
        if "embedding" in self.df.columns:
            emb = torch.tensor(self.df.iloc[idx]['embedding'], dtype=torch.float32).squeeze()
        else:
            emb = None  # No embeddings available
    
        return image, text, category_id, file_path


    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def get_class_id(self, idx: int) -> int:
        """
        Returns the class ID of a specific sample.
        """
        return self.df["category_id"].iloc[idx]

    def show_sample(self, idx: int) -> None:
        """
        Displays a sample image along with its class name and index.
        """
        image, category_id, _, _ = self.__getitem__(idx)
        class_name = self.category_id2label[category_id]

        plt.imshow(image)
        plt.title(f"Class: {class_name}; id: {idx}")
        plt.axis('off')
        plt.show()

    def get_category_idxs(self, category_id: int) -> List[int]:
        """
        Retrieves all indexes for a given category ID.
        """
        return self.df[self.df.category_id == category_id].index.tolist()