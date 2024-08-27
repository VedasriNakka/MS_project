import torch
import cv2
import os
import argparse
import logging

import random
import numpy as np
import albumentations as A
from kornia.morphology import dilation
from albumentations.pytorch import ToTensorV2
from src.core.dataloader_builder import get_dataloader
from src.core.visualization_utils import draw_confusion_matrix, save_image_batch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def compose_transform(transform_types):

    transform_dict = {


        # override 96 with 256.
        #"resize96": A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=1.0),

        "resize224": A.Resize(height=224, width=224, p=1),
        "resize256": A.Resize(height=256, width=256, p=1),
        "randomcrop224": A.RandomCrop(224, 224, p=1.0),     
        
        #"resize256": A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=1.0),
        # scale hyperparam
        "morpho_dilation": A.Morphological(scale=(7, 7), operation="dilation", p=0.5),
        # scale hyperparam
        "morpho_erosion": A.Morphological(scale=(7, 7), operation="erosion", p=0.5),
        "hflip": A.HorizontalFlip(p=0.5),
        "colorjitter": A.ColorJitter(
            brightness=(0.8, 1),
            contrast=(0.8, 1),
            saturation=(0.8, 1),
            hue=(-0.5, 0.5),
            always_apply=False,
            p=0.5,
        ),
        # blur_limit
        "gaussianblur": A.GaussianBlur(
            blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5
        ),
        #  shift, scale, rotate
        "affine": A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5,
        ),
        "invert": A.InvertImg(p=0.5),
        "gray": A.ToGray(p=0.5),
    }


    if True:
        train_composed_transforms = []


        # mandatorily  resize all input images to 256x256
        train_composed_transforms.append(transform_dict["resize256"])

        # construct the transforms list from the input string
        if transform_types != "":
            transform_types = transform_types.split(",")  
            for transform_type in transform_types:
                if transform_type not in transform_dict:
                    raise ValueError(f"Invalid transform_type: {transform_type}")
                train_composed_transforms.append(transform_dict[transform_type])

        else:
            assert False, "Please pass the transform type!"

        #  mandatorily add the normalization transforms
        train_composed_transforms.extend(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    # print(composed_transforms)

    tranform_dict = {
        "train": A.Compose(train_composed_transforms),
        "valid": A.Compose(
            [
                A.Resize(height=256, width=256, p=1),
                A.CenterCrop(height=224, width=224),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [
                A.Resize(height=256, width=256, p=1),
                A.CenterCrop(height=224, width=224),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    }

    return tranform_dict


# Applying transforms to the data
def get_transform(transform_type):
    return compose_transform(transform_type)


def get_dataset(args):
    if args.dataset == "icdar":
        train_directory = (
            "/home/vedasri/datasets/HomerCompTraining_Cropped"  #'imds_small/train'
        )

    elif args.dataset == "alpub":
        train_directory = "/home/vedasri/datasets/alpub_v2/images"  #'imds_small/train'
    else:
        raise NotImplementedError

    return train_directory


def main(args):

    train_directory = get_dataset(args)

    transform_name = args.transform_type
    transforms_dict = get_transform(transform_name)
    print(transforms_dict["train"])
    dataloaders, class_names = get_dataloader(
        train_directory=train_directory,
        transforms_dict=transforms_dict,
        args=args,
        logger=logger,
        shuffle_train=False,
        add_manual_seed=True,
    )

    phase = "train"

    output_dir = os.path.join("../../augmentation_visuals", phase)
    for index, (inputs, labels) in enumerate(dataloaders[phase]):
        # print(labels)

        images = inputs["image"]

        print(torch.mean(images))
        save_image_batch(images, f"{index}_" + transform_name + "_" + phase, output_dir)
        if index == 5:
            break


if __name__ == "__main__":

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    # Set a seed value, for example, 42
    setup_seed(42)

    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform_type", help="transform type", default="", type=str)
    ap.add_argument("--dataset", help="dataset", default="icdar", type=str)
    ap.add_argument("--batch_size", help="batch size", default=64, type=int)

    # args = vars(ap.parse_args())
    args = ap.parse_args()

    main(args)
