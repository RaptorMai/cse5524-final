from tkinter.constants import RAISED

import timm
import torch
from experiment.build_clip_zs_classifier import get_clip_proj, init_fungi_clip
from model.vision_transformer import VisionTransformerPETL
from utils.log_utils import log_model_info
from utils.setup_logging import get_logger

logger = get_logger("PETL_vision")

TUNE_MODULES = ['ft_attn_module', 'ft_mlp_module', 'head', 'vpt', 'ssf_scale', 'ssf_shift', 'lora', 'fact', 'vqt',
                'difffit']

def get_model(params):
    if torch.cuda.is_available():
        params.device = torch.cuda.current_device()
    else:
        raise Exception("No GPU available")

    model = get_base_model(params)

    ##########
    tune_parameters = []
    if params.debug:
        logger.info("Trainable params:")

    if params.bitfit or params.difffit:
        TUNE_MODULES.append('bias')

    if params.ln or params.difffit:
        TUNE_MODULES.append('norm')

    if params.mlp_index:
        if isinstance(params.mlp_index, str):
            params.mlp_index = eval(params.mlp_index)
        for i in params.mlp_index:
            if params.mlp_type == 'fc1':
                TUNE_MODULES.append(str(i) + '.mlp.fc1')
            elif params.mlp_type == 'fc2':
                TUNE_MODULES.append(str(i) + '.mlp.fc2')
            elif params.mlp_type == 'full':
                TUNE_MODULES.append(str(i) + '.mlp.fc1')
                TUNE_MODULES.append(str(i) + '.mlp.fc2')
            else:
                raise NotImplementedError

    if params.attention_index:
        if isinstance(params.attention_index, str):
            params.attention_index = eval(params.attention_index)
        for i in params.attention_index:
            if params.attention_type == 'qkv':
                TUNE_MODULES.append(str(i) + '.attn.qkv')
            elif params.attention_type == 'proj':
                TUNE_MODULES.append(str(i) + '.attn.proj')
            elif params.attention_type == 'full':
                TUNE_MODULES.append(str(i) + '.attn.qkv')
                TUNE_MODULES.append(str(i) + '.attn.proj')
            else:
                raise NotImplementedError

    if params.block_index:
        if isinstance(params.block_index, str):
            params.block_index = eval(params.block_index)
        for i in params.block_index:
            TUNE_MODULES.append('blocks.' + str(i))

    for name, parameter in model.named_parameters():
        if params.full:
            parameter.requires_grad = True
            tune_parameters.append(parameter)
            if params.debug:
                logger.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
        else:
            if any(m in name for m in TUNE_MODULES):
                parameter.requires_grad = True
                tune_parameters.append(parameter)
                if params.debug:
                    logger.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
            else:
                parameter.requires_grad = False

    model_grad_params_no_head = log_model_info(model, logger)

    model = model.cuda(device=params.device)
    return model, tune_parameters, model_grad_params_no_head

def get_base_model(params):
    if params.pretrained_weights == "vit_base_patch16_224_in21k":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_224_in21k_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False, params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_in21k.npz')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_mae":
        model = timm.create_model("vit_base_patch16_224_in21k_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/mae_pretrain_vit_base.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_patch14_dinov2_petl":
        params.patch_size = 14
        model = timm.create_model("vit_base_patch14_dinov2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_14_dinov2.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_patch16_dinov2_petl":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_dinov2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_dinov2.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_large_patch14_dinov2_petl":
        params.patch_size = 14
        model = timm.create_model("vit_large_patch14_dinov2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-L_14_dinov2.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_so400m_patch16_siglip2_petl":
        params.patch_size = 16
        model = timm.create_model("vit_so400m_patch16_siglip2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-SO400M_16_SigLIP2_512.bin')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_large_patch16_siglip2_petl":
        params.patch_size = 16
        model = timm.create_model("vit_large_patch16_siglip2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-L_16_SigLIP2_512.bin')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_patch16_siglip2_petl":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_siglip2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_SigLIP2_512.bin')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == 'vit_base_patch16_clip_224_petl':
        params.patch_size = 16
        model_type = 'ViT-B-16'
        model = timm.create_model("vit_base_patch16_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_clip.bin')

        proj = get_clip_proj(params.device, model_type)
        fc = init_fungi_clip(params.device, model_type)
        model.head = torch.nn.Sequential(*[proj, fc])
        # model.reset_classifier(params.class_num)
    elif params.pretrained_weights == 'vit_large_patch14_clip_224_petl':
        params.patch_size = 14
        model_type = 'ViT-L-14'
        model = timm.create_model("vit_large_patch14_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-L_14_clip.bin')

        proj = get_clip_proj(params.device, model_type)
        fc = init_fungi_clip(params.device, model_type)
        model.head = torch.nn.Sequential(*[proj, fc])
        # model.reset_classifier(params.class_num)
    elif params.pretrained_weights == 'vit_base_patch16_bioclip_224_petl':
        params.patch_size = 16
        model_type = 'ViT-B-16'
        model = timm.create_model("vit_base_patch16_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        model.load_pretrained(
            'pretrained_weights/ViT-B_16_bioclip.bin')

        proj = get_clip_proj(params.device, model_type)
        fc = init_fungi_clip(params.device, model_type)
        model.head = torch.nn.Sequential(*[proj, fc])
        # model.reset_classifier(params.class_num)
    else:
        raise NotImplementedError

    return model
