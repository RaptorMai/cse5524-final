import os
import clip
import torch
import pickle
from utils.setup_logging import get_logger
logger = get_logger("PETL_vision")
from torch import nn

def get_clip_proj(device, model_type):
    save_path = f'pretrained_weights/fungi_{model_type}_clip_proj.pkl'
    if os.path.exists(save_path):
        logger.info(f"load proj from {save_path}")
        with open(save_path, 'rb') as f:
            proj = pickle.load(f)
            proj.to(device)
            return proj
    else:
        if model_type == 'ViT-B-16' or model_type == 'ViT-B-14':
            model, _ = clip.load('ViT-B/16', device, jit=False)
        elif model_type == 'ViT-L-14' or model_type == 'ViT-L-16':
            model, _ = clip.load('ViT-L/14', device, jit=False)
        clip_proj = model.visual.proj
        clip_proj.data = clip_proj.data.float()
        clip_proj = CustomLinear(clip_proj)
        if os.path.dirname(save_path) != '':
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(clip_proj.cpu(), f)
        return clip_proj

def init_fungi_clip(device, model_type):
    save_path = f'pretrained_weights/fungi_{model_type}_clip_head.pkl'
    if os.path.exists(save_path):
        logger.info(f"load head from {save_path}")
        with open(save_path, 'rb') as f:
            fc = pickle.load(f)
            fc.to(device)
            return fc
    else:
        if model_type == 'ViT-B-16' or model_type == 'ViT-B-14':
            model, _ = clip.load('ViT-B/16', device, jit=False)
        elif model_type == 'ViT-L-14' or model_type == 'ViT-L-16':
            model, _ = clip.load('ViT-L/14', device, jit=False)
        logger.info(f"building pred head from {model}")
        model.eval()
        model.to(device)
        with open('data/dataset/fungi_class_set.txt', 'r') as f:
            class_list = [line.strip() for line in f]
        with torch.no_grad():
            texts = clip.tokenize(class_list).to(device)
            zeroshot_weights = model.encode_text(texts).to(device)

            zeroshot_weights *= model.logit_scale.exp()
            zeroshot_weights /= zeroshot_weights.norm(dim=-1, keepdim=True)
            zeroshot_weights = zeroshot_weights.float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        fc = ClassificationHead(normalize=True, weights=zeroshot_weights)
        fc.to(device)
        if os.path.dirname(save_path) != '':
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(fc.cpu(), f)
        return fc

class CustomLinear(nn.Module):
    def __init__(self, existing_parameter):
        super(CustomLinear, self).__init__()
        self.linear = nn.Linear(existing_parameter.shape[0], existing_parameter.shape[1], bias=False)

        # Set the weights to the existing parameter
        self.linear.weight = nn.Parameter(existing_parameter.T)

    def forward(self, x):
        return self.linear(x)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        input_size, output_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.t().clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)