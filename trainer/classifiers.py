import os
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
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as T
import logging

from typing import Sequence, Tuple, Any, Dict, List, Optional, Union

from .ClipWrapper import ClipObject

from PIL import ImageFile

class NNClassifier():
    def __init__(self, train_dataset, device='cuda:0'):
        """
        :param cfg: config object, namespace
        :param train_embeddings: list of C torch arrays of shape [N_C, D] where N_C is the number of training samples
        of class C and D is the dimensionality of the embeddings
        """
        
        self.device = device
        self.index, self.idx2cls = self.build_index(train_dataset)

    def make_prediction(self, embeddings, plot_sim_hist=False, ret_probs=False):
        """
        :param embeddings: torch.Tensor of shape (batch_size, n_channels, height, width)
        :return: probabilities of shape (batch_size, n_classes) computed based on
        the similarity of the embeddings to the class prototypes
        """
        # compute the similarity of each embedding to each prototype
        # embeddings - [N, D], class_prototypes - [C, D]

        similarities, indices = self.index.search(embeddings, 5)
        # get the classes for the indices
        cls = self.idx2cls[indices.squeeze()]
        # get the confidence of the prediction
        conf = similarities
        
        return cls, conf

    def build_index(self, train_dataset):
        idx2cls = train_dataset.df.category_id.values
        # concatenate the embeddings
        embs = np.array(train_dataset.df.embedding.values.tolist(), dtype=np.float32).squeeze()
        # build the index for cosine similarity search
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        return index, idx2cls

class PrototypeClassifier(torch.nn.Module):
    def __init__(self, train_dataset, clip_model=None, device='cuda:0'):
        super().__init__()
        # self.visual_model = clip_model.model.visual
        # self.preprocess = clip_model.preprocess
        # self.randomPreprocess = clip_model.randomPreprocess
        # self.tokenizer = clip_model.tokenizer
        self.device = device
        self.train_dataset = train_dataset

        class_embeddings, _ = self._get_classifier_embeddings(train_dataset)
        self.class_prototypes = torch.nn.Parameter(self.get_prototypes(class_embeddings), requires_grad=False)

    def _get_classifier_embeddings(self, dataset_train):
        class_embeddings = []
        empty_classes = []
        n_classes = min(torch.inf, dataset_train.n_classes)
        for cls in range(n_classes):
            cls_embs = dataset_train.get_embeddings_for_class(cls)
            if len(cls_embs) == 0:
                # if no embeddings for class, use zeros
                empty_classes.append(cls)
                class_embeddings.append(torch.zeros(1, dataset_train.emb_dim))
            else:
                class_embeddings.append(torch.tensor(np.vstack(cls_embs.values)))
        return class_embeddings, empty_classes

    def get_prototypes(self, embeddings):
        return torch.stack([class_embs.mean(dim=0) for class_embs in embeddings])

    def make_prediction(self, embeddings):
        embeddings = embeddings.squeeze(1)
        
        # Normalize the embeddings if they aren't already
        embeddings = F.normalize(embeddings, dim=1, p=2)
        prototypes = F.normalize(self.class_prototypes, dim=1, p=2)
        
        # Compute similarities with proper broadcasting
        # (N, D) @ (C, D).T -> (N, C)
        similarities = torch.matmul(embeddings, prototypes.t())
        
        # Get top-5 predictions and confidences
        topk_values, topk_indices = torch.topk(similarities, 5, dim=1)
        conf = torch.nn.functional.softmax(similarities, dim=1).max(dim=1).values
        
        return topk_indices, conf

class CLIPClassifier(nn.Module):
    def __init__(self, clip: ClipObject):
        super(CLIPClassifier, self).__init__()
        self.visual_model = clip.model.visual
        self.preprocess = clip.preprocess
        self.randomPreprocess = clip.randomPreprocess
        self.tokenizer = clip.tokenizer
        self.device = clip.device
        self.head = None
        self.initialized = False
    
    def add_class_set(self, class_set):
        self.class_set = class_set

    def unfreeze_head(self):
        for param in self.head.parameters():
            param.requires_grad = True

    def init_head(self, class_embedding):
        assert not self.initialized, 'Head already initialized. '
        self.head = nn.Linear(class_embedding.size(1), class_embedding.size(0), bias=True).to(self.device)
        self.head.weight.data = class_embedding.to(self.device)
        self.head.bias.data.zero_()
        self.initialized = True

    def forward(self, images):
        x = self.visual_model(images)
        x = F.normalize(x, dim=-1)
        x = self.head(x)
        return x

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            logging.info(f'Creating directory {os.path.dirname(path)}... ')
            os.makedirs(os.path.dirname(path))
        logging.info(f'Saving classifier to {path}... ')
        torch.save({
            'visual_model_state_dict': self.visual_model.state_dict(),
            'head_state_dict': self.head.state_dict(),
        }, path)
    
    def load(self, path):
        logging.info(f'Loading classifier from {path}... ')
        checkpoint = torch.load(path, map_location=self.device)

        missing_keys, unexpected_keys = self.visual_model.load_state_dict(checkpoint['visual_model_state_dict'], strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        if self.head is None:
            self.head = nn.Linear(checkpoint['head_state_dict']['weight'].shape[1], 
                                  checkpoint['head_state_dict']['weight'].shape[0], 
                                  bias=True).to(self.device)
        self.head.load_state_dict(checkpoint['head_state_dict'])
        logging.info('Classifier loaded successfully')
    
    def extract_img_features(self, image):
        """
        Extract normalized feature embeddings from a given image.

        Args:
            image (PIL.Image.Image): The input image from which to extract features.

        Returns:
            torch.Tensor: A normalized feature embedding vector for the input image.

        Raises:
            ValueError: If the model has not been loaded prior to calling this method.
        """
        if self.visual_model is None:
            raise ValueError('Model not loaded')
        image_tensor_proc = self.preprocess(image)[None]
        features = self.visual_model(image_tensor_proc.to(self.device))
        return self.normalize_embedding(features)
    
    @staticmethod
    def normalize_embedding(embs):
        """
        Normalize the embedding vectors to have unit length.

        Args:
            embs (torch.Tensor): The raw embedding vectors.

        Returns:
            torch.Tensor: L2-normalized embedding vectors.
        """
        return F.normalize(embs.float(), dim=1, p=2)
    
    def make_prediction(self, embeddings):
        embeddings = embeddings.squeeze(1)
        logits = self.head(embeddings)
        cls = torch.argmax(logits, dim=1).cpu()
        conf = torch.nn.functional.softmax(logits, dim=1).max(dim=1).values
        return cls, conf

    def make_logits(self, embeddings):
        """
        Return raw logits for given embeddings (without selecting top-1).
        """
        embeddings = embeddings.squeeze(1)
        logits = self.head(embeddings)
        return logits

def generate_embeddings(dataset, tp, model):
    idxs = np.arange(len(dataset))
    im_names, embs = [], []
    print("Generating embeddings...")
    for idx in idxs:
        img, text, label, file_path = dataset[idx]

        with torch.no_grad():
            img = Image.open(img).convert('RGB')
            feat = model.extract_img_features(img)

        im_names.append(os.path.basename(file_path))
        embs.append(feat.detach().cpu().numpy())

    embeddings = pd.DataFrame({'filename': im_names, 'embedding': embs})

    return embeddings

def build_prototype_classifier(clip_object, train_dataset):
    clip_object.model.eval()
    clip_object.modify_lora_mergeFactor(merge_factor=0.0)
    embeddings = generate_embeddings(train_dataset, 'train', clip_object)
    clip_object.modify_lora_mergeFactor(merge_factor=1.0)
    train_dataset.add_embeddings(embeddings)
    Proclassifier = PrototypeClassifier(train_dataset)
    classifier = CLIPClassifier(clip_object)
    classifier.init_head(Proclassifier.class_prototypes)
    del Proclassifier
    del embeddings
    return classifier

def build_CLIPclassifier(clip_object, class_set):
    classifier = CLIPClassifier(clip_object)
    classifier.add_class_set(class_set)
    class_embedding = clip_object.extract_text_features(class_set)
    classifier.init_head(class_embedding)
    return classifier
