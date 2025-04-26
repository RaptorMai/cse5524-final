import torch
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.nn import functional as F
import torch.nn as nn
import torch
import sys
import open_clip
import logging

from .open_clip import create_model_and_transforms

class ClipObject(torch.nn.Module):
    def __init__(self, model_cfg, device):
        super().__init__()
        if model_cfg['fine_tune'] == 'LoRA':
            self.model, _, self.preprocess = create_model_and_transforms(
                model_cfg['model_version'],
                pretrained=model_cfg['pretrain_dataset'],
                device=device,
                custom=True,
                LoRA_dim=model_cfg['LoRA_dim'],
            )
        else:
            self.model, _, self.preprocess = create_model_and_transforms(
                model_cfg['model_version'],
                pretrained=model_cfg['pretrain_dataset'],
                device=device,
                custom=False,
            )
        # self.randomPreprocess = transforms.Compose([
        #     transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        #     transforms.RandomCrop(size=(224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
        #                        std=(0.26862954, 0.26130258, 0.27577711))
        # ])
        self.randomPreprocess = T.Compose([
            T.RandomResizedCrop(size=(224,224)),      # random crop then resize (common in few-shot miniImageNet)
            T.RandomHorizontalFlip(p=0.5),         # 50% chance to flip
            # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # random color jitter
            T.RandomRotation(15),                  # random rotation between -15 and 15 degrees
            T.ToTensor(),                        # convert PIL image to tensor for model
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                std=(0.26862954, 0.26130258, 0.27577711))  # (optional) normalize if using pre-trained models
        ])
        if model_cfg['bioclip_tokenizer']:
            self.tokenizer = open_clip.get_tokenizer(model_cfg['bioclip_tokenizer'])
        else:
            self.tokenizer = open_clip.get_tokenizer(model_cfg['model_version'])
        
        # Load bioclip weights if specified
        if 'bioclip_ckpt' in model_cfg and model_cfg['bioclip_ckpt']:
            self.load_weights(model_cfg['bioclip_ckpt'], device)
            
        self.logit_scale = self.model.logit_scale
        self.device = device

    def load_weights(self, weights_path, device):
        """Load pre-trained weights into the custom model."""
        print(f'Loading pre-trained weights from {weights_path}')
        state_dict = torch.load(weights_path, map_location=device)
        
        # Modify state dict keys to match the custom model structure
        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle the transformer and visual transformer key mappings
            if k.startswith('visual.'):
                new_key = k
            elif k.startswith('text.'):
                new_key = k.replace('text.transformer', 'transformer')
            else:
                new_key = k
            new_state_dict[new_key] = v

        # Load weights with strict=False to allow for architecture differences
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        
        print("Pre-trained weights loaded successfully")

    def merge_weights(self, weights_path):
        """Merge weights into the current model."""
        print(f'Merging weights from {weights_path}')
        model = torch.load(weights_path, map_location=self.device)
        state_dict = model['state_dict']
        
        # Modify state dict keys to match the custom model structure
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('model.clip_model.', '')
            # Handle the transformer and visual transformer key mappings
            if k.startswith('visual.'):
                new_key = k
            elif k.startswith('text.'):
                new_key = k.replace('text.transformer', 'transformer')
            else:
                new_key = k
            new_state_dict[new_key] = v

        # Merge weights with a 1:1 ratio
        for k, v in new_state_dict.items():
            if k in self.model.state_dict():
                new_state_dict[k] = (self.model.state_dict()[k] + v) / 2
            else:
                new_state_dict[k] = v

        # Load weights with strict=False to allow for architecture differences
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        
        print("Weights merged successfully")

    def extract_text_features(self, text):
        """
        Extract normalized feature embeddings from a given text.

        Args:
            text (str): The input text from which to extract features.

        Returns:
            torch.Tensor: A normalized feature embedding vector for the input text.

        Raises:
            ValueError: If the model has not been loaded prior to calling this method.
        """
        if self.model is None:
            raise ValueError('Model not loaded')
        text_tokens = self.tokenizer(text).to(self.device)
        features = self.model.encode_text(text_tokens)
        return self.normalize_embedding(features)

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
        if self.model is None:
            raise ValueError('Model not loaded')
        image_tensor_proc = self.preprocess(image)[None]
        features = self.model.encode_image(image_tensor_proc.to(self.device))
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

    def print_parameters(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, images, texts):
        assert self.model is not None
        # if text==None:
        #     return self.model.encode_image(images)
        # else:
        
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text

    def half(self):
        self.model = self.model.half()
    
    def floatMode(self):
        self.model = self.model.float()

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        # Save both model state and configuration
        torch.save(self.model.state_dict(), filename)

    def freeze(self):
        print("Turning off gradients in both the image and the text encoder")
        self.all_elements = 0
        # for name, param in self.model.named_parameters():
        for param in self.model.parameters():
            param.requires_grad = False
            self.all_elements += param.numel()
    
    def freeze_backbone(self):
        # Freeze the backbone parameters
        for param in self.model.visual.parameters():
            param.requires_grad = False

        for param in self.model.transformer.parameters():
            param.requires_grad = False
    
    def unfreeze_param_fft(self):
        self.all_elements = 0
        for param in self.model.parameters():
            param.requires_grad = False
            self.all_elements += param.numel()
        # Unfreeze the visual for fft
        for param in self.model.visual.parameters():
            param.requires_grad = True

        print("Printing unfrozen parameters: ")
        unfrozen_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad == True]
        total_elements = 0
        for name, param in unfrozen_params:
            num_elements = param.numel()
            total_elements += num_elements
            print(f"Parameter name: {name}, Shape: {param.shape}, Number of elements: {num_elements}")
        print(f"Total number of elements in tunable parameters: {total_elements}")
        self.total_elements = total_elements
        print(f"Percentage of tunable parameters: {(self.total_elements/self.all_elements * 100):.12f}%")

    def modify_lora_mergeFactor(self, merge_factor):
        for block in self.model.visual.transformer.resblocks:
            if not hasattr(block.attn, 'lora'):
                print(f"Block {block} does not have LoRA parameters.")
                continue
            block.attn.lora.merge_factor = merge_factor
        print(f"LoRA merge factor set to {merge_factor}")

    def unfreeze_param_LoRA(self):
        self.all_elements = 0
        for param in self.model.parameters():
            param.requires_grad = False
            self.all_elements += param.numel()

        # Unfreeze LoRA parameters
        for block in self.model.visual.transformer.resblocks:
            for param in block.attn.lora.lora_a.parameters():
                param.requires_grad = True
            for param in block.attn.lora.lora_b.parameters():
                param.requires_grad = True

        print("Printing unfrozen parameters: ")
        unfrozen_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad == True]
        total_elements = 0
        for name, param in unfrozen_params:
            num_elements = param.numel()
            total_elements += num_elements
            print(f"Parameter name: {name}, Shape: {param.shape}, Number of elements: {num_elements}")
        print(f"Total number of elements in tunable parameters: {total_elements}")
        self.total_elements = total_elements
        print(f"Percentage of tunable parameters: {(self.total_elements/self.all_elements * 100):.12f}%")

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")
