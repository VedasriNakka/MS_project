"""

python datatransform_builder_v2.py --transform_type resize256,affine

"""

from torchvision import datasets, models, transforms
import torch
import cv2
import os
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import random
import numpy as np
import albumentations as A
from kornia.morphology import dilation
from albumentations.pytorch import ToTensorV2


import torchvision.transforms as transforms


def compose_transform(transform_types):

    transform_dict = {
        "default": A.Resize(height=256, width=256, p=1.0),
        "resize256": A.Resize(height=256, width=256, p=1),
        "randomcrop224": A.RandomCrop(224, 224, p=1.0),     
        "randomcrop198": A.RandomCrop(198, 198, p=1.0),     
        "randomcrop128": A.RandomCrop(128, 128, p=1.0),     
        "randomcrop180": A.RandomCrop(180, 180, p=1.0),     

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

    composed_transforms = []

    
    # mandatorily  resize all input images to 256x256
    composed_transforms.append(transform_dict["resize256"])


    # add all interim transforms
    if transform_types != "":
        transform_types = transform_types.split(
            ","
        )  # Split the string into a list of transform types
        for transform_type in transform_types:
            if transform_type not in transform_dict:
                raise ValueError(f"Invalid transform_type: {transform_type}")
            composed_transforms.append(transform_dict[transform_type])

    else:
        assert False, "Not implemented!"
  
    RESIZE_DIM = 96

    # add all final preprocessing transforms
    composed_transforms.extend(
        [

            # we resize all images to RESIZE_DIM
            A.Resize(height=RESIZE_DIM, width=RESIZE_DIM, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    transform_type = A.Compose(composed_transforms)
    test_or_val_transforms =  A.Compose(
            [
                A.Resize(height=256, width=256, p=1),
                A.CenterCrop(height=224, width=224),
                
                A.Resize(height=RESIZE_DIM, width=RESIZE_DIM, p=1),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
    )
    #return {"train": transform_type, "valid": test_or_val_transforms, "test": test_or_val_transforms}
    return {"train": transform_type, "valid": transform_type, "test": transform_type}


# Applying transforms to the data
def get_transform(transform_type):
    return compose_transform(transform_type)


def get_dataset(args):
    if args.dataset == "icdar":
        train_directory = "/home/vedasri/datasets/HomerCompTraining_Cropped"

    elif args.dataset == "alpub":
        train_directory = "/home/vedasri/datasets/alpub_v2/images"
    else:
        raise NotImplementedError

    return train_directory


if __name__ == "__main__":

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    setup_seed(42)

    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform_type", help="transform type", default="", type=str)
    ap.add_argument("--dataset", help="dataset", default="icdar", type=str)
    ap.add_argument("--batch_size", help="batch size", default=64, type=int)
    args = ap.parse_args()
