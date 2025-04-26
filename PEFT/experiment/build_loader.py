import torch
from data.dataset.fungi import FungiTastic, BalancedClassSampler
from PIL import Image
import torchvision.transforms as T
import random
from PIL import ImageEnhance, ImageFilter
from torchvision.transforms import InterpolationMode

class SolarizeCustom(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        return T.functional.solarize(img, threshold=self.threshold)

class RandomSharpness:
    def __init__(self, p=0.5, factor_range=(0.5, 2.0)):
        self.p = p
        self.factor_range = factor_range
    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            return ImageEnhance.Sharpness(img).enhance(factor)
        return img

def get_dataset(data, params, logger):
    dataset_train, dataset_val, dataset_test = None, None, None
    trainsform_val = T.Compose([
            # T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))])
    transform_train = T.Compose([
                # T.RandomResizedCrop(crop_size, scale=(0.4, 1.0)),  # e.g., "global" style crop
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=30),

                # --- OPTION 1: AutoAugment ---
                # T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
                #
                # --- OPTION 2: RandAugment (with chosen hyperparams) ---
                # T.RandAugment(num_ops=3, magnitude=9),
                #
                # --- OPTION 3: TrivialAugment ---
                # T.TrivialAugmentWide(),

                T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=23)], p=1.0),
                T.RandomApply([SolarizeCustom(threshold=128)], p=0.2),
                T.RandomApply([RandomSharpness(p=1.0)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                # Random Erasing goes last (applies on Tensor)
                T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')])

    # train_transform_list = list(transform_train.transforms)
    if params.dataAug == 'TrivialAugment':
        transform_train.transforms.insert(4, T.TrivialAugmentWide())
    elif params.dataAug == 'RandAugment':
        transform_train.transforms.insert(4, T.RandAugment(num_ops=3, magnitude=9))
    elif params.dataAug == 'AutoAugment':
        transform_train.transforms.insert(4, T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
    else:
        raise NotImplementedError(f"dataAug {params.dataAug} not implemented")

    if params.img_size != 'whole':
        crop_size = int(params.img_size)
        transform_train.transforms.insert(0, T.RandomResizedCrop(crop_size, scale=(0.4, 1.0), interpolation=InterpolationMode.BICUBIC))
        trainsform_val.transforms.insert(0, T.CenterCrop(crop_size))

    data_root = params.data_root
    img_folder = params.img_quality
    dataset_train = FungiTastic(data_root, split='train', usage='training', transform_train=transform_train, transform_val=trainsform_val, img_folder=img_folder)
    dataset_val = FungiTastic(data_root, split='val', usage='testing', transform_train=transform_train, transform_val=trainsform_val, img_folder=img_folder)
    dataset_test = FungiTastic(data_root, split='test', usage='testing', transform_train=transform_train, transform_val=trainsform_val, img_folder=img_folder)
    return dataset_train, dataset_val, dataset_test

def get_loader(params, logger):
    if 'test_data' in params:
        dataset_train, dataset_val, dataset_test = get_dataset(params.test_data, params, logger)
    else:
        dataset_train, dataset_val, dataset_test = get_dataset(params.data, params, logger)

    if isinstance(dataset_train, list):
        train_loader, val_loader, test_loader = [], [], []
        for i in range(len(dataset_train)):
            tmp_train, tmp_val, tmp_test = gen_loader(params, dataset_train[i], dataset_val[i], None)
            train_loader.append(tmp_train)
            val_loader.append(tmp_val)
            test_loader.append(tmp_test)
    else:
        train_loader, val_loader, test_loader = gen_loader(params, dataset_train, dataset_val, dataset_test)

    logger.info("Finish setup loaders")
    return train_loader, val_loader, test_loader

def gen_loader(params, dataset_train, dataset_val, dataset_test):
    train_loader, val_loader, test_loader = None, None, None
    if params.debug:
        num_workers = 1
    else:
        num_workers = 4
    if dataset_train is not None:
        # Create a balanced sampler
        if params.data_size == 'full':
            balanced_sampler = BalancedClassSampler(dataset_train)
        else:
            balanced_sampler = None
            dataset_train.balance_df() ## This will only use the minimal balanced dataset
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=params.batch_size,
            sampler=balanced_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    if dataset_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    if dataset_test is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    return train_loader, val_loader, test_loader
