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

import random
from torch.utils.data import Sampler

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import transforms as tfms

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union, Iterator
import importlib
from PIL import ImageFile
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes
from math import pi
import xgboost as xgb

ImageFile.LOAD_TRUNCATED_IMAGES = True

def geo_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return pd.DataFrame({'x': x, 'y': y, 'z': z})

def month_to_cyclical(month):
    radians = 2 * pi * (month % 12) / 12
    return pd.DataFrame({
        'month_sin': np.sin(radians),
        'month_cos': np.cos(radians)
    })

class FungiTastic(torch.nn.Module):
    """
    Dataset class for the FewShot subset of the Danish Fungi dataset (size 300, closed-set).

    This dataset loader supports training, validation, and testing splits, and provides
    convenient access to images, class IDs, and file paths. It also supports optional
    image transformations.
    """

    
    SPLIT2STR = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
    IMG_DIRS = ['300p', '500p', '720p', 'fullsize']

    def __init__(self, root: str, split: str = 'val', usage='training', transform_train=None, transform_val=None, img_folder='fullsize'):
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
        self.img_folder = img_folder
        self.df = self._get_df(usage, root, split, img_folder=img_folder)
        self.mode = usage
        self.preprocess_val = transform_val

        self.preprocess_train =  transform_train

        # if self.split != 'test':
        #     self.prompt_set = self.get_prompts() 

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
    
    def set_mode(self, mode: str):
        """
        Sets the mode for the dataset. This is used to control the behavior of the dataset
        during training and evaluation.

        Args:
            mode (str): The mode to set. Can be 'train', 'val', or 'test'.
        """
        assert mode in ['train', 'eval', 'test'], "Mode must be one of ['train', 'val', 'test']."
        self.mode = mode

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
    def _get_df(usage: str, data_path: str, split: str, img_folder='fullsize') -> pd.DataFrame:
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
            lambda x: os.path.join(data_path, "FungiTastic-FewShot", split, img_folder, x)
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
            category_id = -1

        image = Image.open(file_path).convert("RGB")
        
        # if self.split != 'test':
        #     text = self.prompt_set[idx]
        # else:
        #     text = 'None'

        if self.mode != 'train':
            image = self.preprocess_val(image)
        else:
            image = self.preprocess_train(image)

        # Check if embeddings exist
        if "embedding" in self.df.columns:
            emb = torch.tensor(self.df.iloc[idx]['embedding'], dtype=torch.float32).squeeze()
        else:
            emb = None  # No embeddings available
    
        return image, category_id

    def merge(self, dataset: 'FungiTastic', mode: str = 'train'):
        """
        Merges another dataset into the current dataset.

        Args:
            dataset (FungiTastic): The dataset to merge.
            mode (str): The mode to set for the merged dataset. Can be 'train', 'val', or 'test'.
        """
        assert isinstance(dataset, FungiTastic), "Dataset must be an instance of FungiTastic."
        self.df = pd.concat([self.df, dataset.df], ignore_index=True)
        self.mode = mode
        self.n_classes = len(self.df['category_id'].unique())
        self.category_id2label = {
            k: v[0] for k, v in self.df.groupby('category_id')['scientificName'].unique().to_dict().items()
        }
        self.label2category_id = {
            v: k for k, v in self.category_id2label.items()
        }
        self.get_prompts()
    
    def prepare_xg(self, train_feature_cols: List[str] = None):
        target = 'scientificName'
        df = self.df.copy()
        substrate_counts = df['substrate'].value_counts()
        def map_substrate(sub):
            if pd.isna(sub):
                return 'missing'
            elif substrate_counts.get(sub, 0) >= 100:
                return sub
            elif substrate_counts.get(sub, 0) >= 20:
                return 'other_common'
            else:
                return 'rare'
        # Spatial + temporal features
        df['substrate_grouped'] = df['substrate'].apply(map_substrate)
        df[['x', 'y', 'z']] = geo_to_cartesian(df['latitude'], df['longitude'])
        df[['month_sin', 'month_cos']] = month_to_cyclical(df['month'])
        # Final categorical handling
        categorical_cols = ['habitat', 'metaSubstrate', 'landcover', 'biogeographicalRegion', 'substrate_grouped']
        low_card = [col for col in categorical_cols if df[col].nunique() <= 32]
        df = pd.get_dummies(df, columns=low_card)
        df = df.fillna(-999)
        
        # species_encoder = LabelEncoder()
        # df[target] = species_encoder.fit_transform(df[target])
        if train_feature_cols is not None:
            feature_cols = [col for col in df.columns if col != target and ptypes.is_numeric_dtype(df[col])]
            X = df[feature_cols].fillna(-999)
            # feature_cols = train_feature_cols
            # X = df.reindex(columns=feature_cols, fill_value=-999).fillna(-999)   # add missing & drop extras
        else:
            feature_cols = [col for col in df.columns if col != target and ptypes.is_numeric_dtype(df[col])]
            X = df[feature_cols].fillna(-999)
        if self.split != 'test':
            self.dtrain = xgb.DMatrix(X)
        else:
            self.dtest = xgb.DMatrix(X)
        return feature_cols, X

    def balance_df(self):
        """
        Balances the dataset by ensuring each class has an equal number of samples.
        """
        # min_samples = self.df['category_id'].value_counts().min()
        balanced_df = self.df.groupby('category_id').apply(lambda x: x.sample(4) if len(x)>=4 else x.sample(len(x))).reset_index(drop=True)
        self.df = balanced_df
        self.n_classes = len(self.df['category_id'].unique())
        self.category_id2label = {
            k: v[0] for k, v in self.df.groupby('category_id')['scientificName'].unique().to_dict().items()
        }
        self.label2category_id = {
            v: k for k, v in self.category_id2label.items()
        }
        self.get_prompts()

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

class BalancedClassSampler(Sampler):
    """
    Samples elements uniformly from all classes to ensure balanced representation.
    
    This sampler first randomly selects a category_id, then randomly samples an image
    from that category, ensuring even rare classes appear frequently in training.
    
    Args:
        dataset: Dataset with a get_category_idxs method
        num_samples: Number of samples per epoch (defaults to dataset length)
        replacement: Whether to sample with replacement
    """
    
    def __init__(
        self, 
        dataset, 
        num_samples: Optional[int] = None,
        replacement: bool = True
    ):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
        
        # Build dictionary mapping each category_id to its sample indices
        self.class_indices: Dict[int, List[int]] = {}
        for category_id in dataset.df['category_id'].unique():
            self.class_indices[category_id] = dataset.get_category_idxs(category_id)
            
        self.class_list = list(self.class_indices.keys())

    def reconstruct(self, dataset):
        del self.class_indices, self.class_list
        self.__init__(dataset)

    def __iter__(self) -> Iterator[int]:
        # Sample with replacement: continue drawing samples until we have enough
        count = 0
        while count < self.num_samples:
            # Sample a class uniformly at random
            class_id = random.choice(self.class_list)
            
            # Sample an instance from this class
            idx = random.choice(self.class_indices[class_id])
            
            yield idx
            count += 1

    def __len__(self) -> int:
        return self.num_samples