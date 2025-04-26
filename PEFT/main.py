import argparse
from experiment.run import basic_run
from utils.setup_logging import get_logger, setup_logging
from utils.misc import set_seed
import time

logger = get_logger("PETL_vision")

def main():
    args = setup_parser().parse_args()
    set_seed(args.random_seed)
    start = time.time()
    basic_run(args)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')

def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision')

    ########################Pretrained Model#########################
    parser.add_argument('--pretrained_weights', type=str, default='vit_base_patch14_dinov2_petl',
                        choices=['vit_base_patch14_dinov2_petl', 'vit_large_patch14_dinov2_petl', 'vit_base_patch16_dinov2_petl', 
                        'vit_large_patch14_clip_224_petl', 'vit_base_patch16_clip_224_petl', 'vit_base_patch16_bioclip_224_petl', 'vit_so400m_patch16_siglip2_petl', 
                        'vit_large_patch16_siglip2_petl', 'vit_base_patch16_siglip2_petl'], 
                        help='pretrained weights name')
    parser.add_argument('--img_size', default='224',
                        choices=['224', '384', '512', '672' '768', '896','whole'],
                        type=str,
                        help='Image size (default: %(default)s)')
    parser.add_argument('--drop_path_rate', default=0.1,
                        type=float,
                        help='Drop Path Rate (default: %(default)s)')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'swin'],
                        help='pretrained model name')
    ########################XGBoost#########################
    parser.add_argument('--xgboost_path', default='xg/xgb_re_noZero.json',
                        type=str,
                        help='Path to the xgboost model. (default: %(default)s)')

    ########################Optimizer Scheduler#########################
    parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--lr', default=0.001,
                        type=float,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--epoch', default=100,
                        type=int,
                        help='The number of total epochs used. (default: %(default)s)')
    parser.add_argument('--warmup_epoch', default=5,
                        type=int,
                        help='warnup epoch in scheduler. (default: %(default)s)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='lr_min for scheduler (default: %(default)s)')
    parser.add_argument('--warmup_lr_init', type=float, default=1e-6,
                        help='warmup_lr_init for scheduler (default: %(default)s)')
    parser.add_argument('--batch_size', default=8,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--test_batch_size', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight_decay (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum used in sgd (default: %(default)s)')
    parser.add_argument('--early_patience', type=int, default=101,
                        help='early stop patience (default: %(default)s)')

    ########################Loss#########################
    parser.add_argument('--loss', default='seesaw_loss', choices=['cross_entropy', 'seesaw_loss', 'focal_loss', 'siglip2_loss'],
                        type=str,
                        help='Loss function (default: %(default)s)')

    ########################Data#########################
    parser.add_argument('--data', default="fungi",
                        help='data name. (default: %(default)s)')
    parser.add_argument('--data_root', default="/research/nfs_chao_209/fungi-clef-2025",
                        help='Path to the dataset. (default: %(default)s)')
    parser.add_argument('--data_size', default='full',
                        type=str,
                        choices=['full', 'part'],
                        help='Data size (default: %(default)s)')
    parser.add_argument('--crop_size', default=224,
                        type=int,
                        help='Crop size of the input image (default: %(default)s)')
    parser.add_argument('--final_run', default=True,
                        type=bool,
                        help='If final_run is True, use train+val as train data else, use train only')
    parser.add_argument('--normalized', action='store_false',
                        help='If imagees are normalized using ImageNet mean and variance ')
    parser.add_argument('--class_num', default=2427,
                        type=int,
                        help='Number of classes (default: %(default)s)')
    parser.add_argument('--dataAug', default='TrivialAugment',
                        choices=['RandAugment', 'TrivialAugment', 'CutMix', 'Mixup', 'AutoAugment', 'Normal'],
                        help='data augmentation method')
    parser.add_argument('--img_quality', default='fullsize',
                        choices=['fullsize', '300p', '500p', '720p'],
                        help='img quality')

    ########################PETL#########################
    parser.add_argument('--ft_attn_module', default=None, choices=['adapter', 'convpass', 'repadapter'],
                        help='Module used to fine-tune attention module. (default: %(default)s)')
    parser.add_argument('--ft_attn_mode', default='parallel',
                        choices=['parallel', 'sequential_after', 'sequential_before'],
                        help='fine-tune mode for attention module. (default: %(default)s)')
    parser.add_argument('--ft_attn_ln', default='before',
                        choices=['before', 'after'],
                        help='fine-tune mode for attention module before layer norm or after. (default: %(default)s)')

    parser.add_argument('--ft_mlp_module', default=None, choices=['adapter', 'convpass', 'repadapter'],
                        help='Module used to fine-tune mlp module. (default: %(default)s)')
    parser.add_argument('--ft_mlp_mode', default='parallel',
                        choices=['parallel', 'sequential_after', 'sequential_before'],
                        help='fine-tune mode for mlp module. (default: %(default)s)')
    parser.add_argument('--ft_mlp_ln', default='before',
                        choices=['before', 'after'],
                        help='fine-tune mode for attention module before layer norm or after. (default: %(default)s)')
    ########################AdaptFormer/Adapter#########################
    parser.add_argument('--adapter_bottleneck', type=int, default=0,
                        help='adaptformer bottleneck middle dimension. (default: %(default)s)')
    parser.add_argument('--adapter_init', type=str, default='lora_kaiming',
                        choices=['lora_kaiming', 'xavier', 'zero', 'lora_xavier'],
                        help='how adapter is initialized')
    parser.add_argument('--adapter_scaler', default=0.1,
                        help='adaptformer scaler. (default: %(default)s)')

    ########################ConvPass#########################
    parser.add_argument('--convpass_xavier_init', action='store_true',
                        help='whether apply xavier_init to the convolution layer in ConvPass')
    parser.add_argument('--convpass_bottleneck', type=int, default=8,
                        help='convpass bottleneck middle dimension. (default: %(default)s)')
    parser.add_argument('--convpass_init', type=str, default='lora_xavier',
                        choices=['lora_kaiming', 'xavier', 'zero', 'lora_xavier'],
                        help='how convpass is initialized')
    parser.add_argument('--convpass_scaler', default=10, type=float,
                        help='ConvPass scaler. (default: %(default)s)')

    ########################VPT#########################
    parser.add_argument('--vpt_mode', type=str, default=None, choices=['deep', 'shallow'],
                        help='VPT mode, deep or shallow')
    parser.add_argument('--vpt_num', default=10, type=int,
                        help='Number of prompts (default: %(default)s)')
    parser.add_argument('--vpt_layer', default=None, type=int,
                        help='Number of layers to add prompt, start from the last layer (default: %(default)s)')
    parser.add_argument('--vpt_dropout', default=0.1, type=float,
                        help='VPT dropout rate for deep mode. (default: %(default)s)')

    ########################SSF#########################
    parser.add_argument('--ssf', action='store_true',
                        help='whether turn on Scale and Shift the deep Features (SSF) tuning')

    ########################lora_kaiming#########################
    parser.add_argument('--lora_bottleneck', type=int, default=0,
                        help='lora bottleneck middle dimension. (default: %(default)s)')

    ########################FacT#########################
    parser.add_argument('--fact_dim', type=int, default=8,
                        help='FacT dimension. (default: %(default)s)')
    parser.add_argument('--fact_type', type=str, default=None, choices=['tk', 'tt'],
                        help='FacT method')
    parser.add_argument('--fact_scaler', type=float, default=1.0,
                        help='FacT scaler. (default: %(default)s)')

    ########################repadapter#########################
    parser.add_argument('--repadapter_bottleneck', type=int, default=8,
                        help='repadapter bottleneck middle dimension. (default: %(default)s)')
    parser.add_argument('--repadapter_init', type=str, default='lora_xavier',
                        choices=['lora_xavier', 'lora_kaiming', 'xavier', 'zero'],
                        help='how repadapter is initialized')
    parser.add_argument('--repadapter_scaler', default=1, type=float,
                        help='repadapter scaler. (default: %(default)s)')
    parser.add_argument('--repadapter_group', type=int, default=2,
                        help='repadapter group')

    ########################BitFit#########################
    parser.add_argument('--bitfit', action='store_true',
                        help='whether turn on BitFit')

    ########################VQT#########################
    parser.add_argument('--vqt_num', default=0, type=int,
                        help='Number of query prompts (default: %(default)s)')
    parser.add_argument('--vqt_dropout', default=0.1, type=float,
                        help='VQT dropout rate for deep mode. (default: %(default)s)')

    ########################MLP#########################
    parser.add_argument('--mlp_index', default=None, type=int, nargs='+',
                        help='indexes of mlp to tune (default: %(default)s)')
    parser.add_argument('--mlp_type', type=str, default='full',
                        choices=['fc1', 'fc2', 'full'],
                        help='how mlps are tuned')

    ########################Attention#########################
    parser.add_argument('--attention_index', default=None, type=int, nargs='+',
                        help='indexes of attention to tune (default: %(default)s)')
    parser.add_argument('--attention_type', type=str, default='full',
                        choices=['qkv', 'proj', 'full'],
                        help='how attentions are tuned')

    ########################LayerNorm#########################
    parser.add_argument('--ln', action='store_true',
                        help='whether turn on LayerNorm fit')

    ########################DiffFit#########################
    parser.add_argument('--difffit', action='store_true',
                        help='whether turn on DiffFit')

    ########################full#########################
    parser.add_argument('--full', action='store_true',
                        help='whether turn on full finetune')

    ########################block#########################
    parser.add_argument('--block_index', default=None, type=int, nargs='+',
                        help='indexes of block to tune (default: %(default)s)')

    ########################domain generalization#########################
    parser.add_argument('--generalization_test', type=str, default='a',
                        choices=['v2', 's', 'a'],
                        help='domain generalization test set for imagenet')
    parser.add_argument('--merge_factor', default=1, type=float,
                        help='merge factor')
    ########################Misc#########################
    parser.add_argument('--gpu_num', default=1,
                        type=int,
                        help='Number of GPU (default: %(default)s)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode to show more information (default: %(default)s)')
    parser.add_argument('--random_seed', default=42,
                        type=int,
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--eval_freq', default=30,
                        type=int,
                        help='eval frequency(epoch) testset (default: %(default)s)')
    parser.add_argument('--store_ckp', default=True,
                        type=bool,
                        help='whether store checkpoint')
    parser.add_argument('--final_acc_hp', action='store_false',
                        help='if true, use the best acc during all epochs as criteria to select HP, if false, use the acc at final epoch as criteria to select HP ')
    parser.add_argument('--run_method', default='trainEval',
                        choices=['train_only', 'trainEval', 'test'],
                        help='run method (default: %(default)s)')
    parser.add_argument('--test_method', default='prototype',
                        choices=['classifier', 'k-NN', 'prototype'],
                        help='test method (default: %(default)s)')
    return parser

if __name__ == '__main__':
    main()
