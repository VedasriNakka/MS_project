from torchvision import datasets, models, transforms
import torch
import cv2
import os
import argparse
from src.core.dataloader_builder import get_dataloader
from src.core.visualization_utils import draw_confusion_matrix, save_image_batch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import random
import numpy as np


def get_resized_transform(hflip=False):

    return {
        "train": transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_resized_hflip_transform():

    return {
        "train": transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_default_transform():
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_center_transform():
    return {
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        "train": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_center_hflip_transform():
    return {
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        "train": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_color_jitter():
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return {
        "train": transforms.Compose(
            [
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.Resize(size=256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_morphology():
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    morpho_operation = transforms.ElasticTransform(alpha=250.0)
    return {
        "train": transforms.Compose(
            [
                transforms.RandomApply([color_jitter], p=0.5),
                # transforms.RandomApply([morpho_operation], p=0.5),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_transform_conditions():
    return [
        "default",
        "resize_224",
        "resize_224_hflip",
        "center_crop",
        "center_crop_hflip" "color_jitter",
        "thinning",
        "morphology",
    ]


# Applying transforms to the data
def get_transform(transform_type):
    if transform_type == "default":
        return get_default_transform()

    elif transform_type == "resize_224":
        return get_resized_transform()

    elif transform_type == "resize_224_hflip":
        return get_resized_hflip_transform(hflip=True)

    elif transform_type == "center_crop":
        return get_center_transform()

    elif transform_type == "center_crop_hflip":
        return get_center_transform()

    elif transform_type == "color_jitter":
        return get_color_jitter()

    elif transform_type == "thinning":
        return get_thinning()

    elif transform_type == "morphology":
        return get_morphology()

    else:
        raise ValueError(f"Invalid transform_type: {transform_type}.")


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

    tranform_types = get_transform_conditions()

    for transform_name in tranform_types:

        print(transform_name)

        if (transform_name == args.transform_type) or (args.transform_type == "all"):

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

                print(torch.mean(inputs))
                save_image_batch(
                    inputs, f"{index}_" + transform_name + "_" + phase, output_dir
                )
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
    ap.add_argument("--transform_type", help="transform type", default="all", type=str)
    ap.add_argument("--dataset", help="dataset", default="icdar", type=str)
    ap.add_argument("--batch_size", help="batch size", default=64, type=int)

    # args = vars(ap.parse_args())
    args = ap.parse_args()

    main(args)
