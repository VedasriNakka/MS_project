"""

python src/core/datatransform_builder_v3.py --transform_type randomcrop224,gaussianblur,gray

"""

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
import albumentations as A
from kornia.morphology import dilation
from albumentations.pytorch import ToTensorV2


import torchvision.transforms as transforms


def compose_transform(transform_types, **kwargs):

    # print(dilation_args)
    # exit(0)

    p = 1.0

    transform_dict = {
        # "resize256": A.RandomResizedCrop(size=(256, 256),
        #                                  scale=(0.8, 1.0),
        #                                  p=1.0),
        "resize256": A.Resize(height=256, width=256, p=1),
        "randomcrop224": A.RandomCrop(224, 224, p=1.0),    #77%
        # scale hyperparam
        "morpho_dilation": A.Morphological(
            p=p, operation="dilation", **kwargs["dilation_args"]
        ),
        # scale hyperparam
        "morpho_erosion": A.Morphological(
            p=p, operation="erosion", **kwargs["erosion_args"]
        ),
        "hflip": A.HorizontalFlip(p=p),
        "colorjitter": A.ColorJitter(
            always_apply=False, p=p, **kwargs["colorjitter_args"]
        ),
        # blur_limit
        "gaussianblur": A.GaussianBlur(
            # blur_limit=(3, 7),
            sigma_limit=0,
            always_apply=False,
            p=p,
            **kwargs["gaussian_args"],
        ),
        #  shift, scale, rotate
        "affine": A.ShiftScaleRotate(
            border_mode=cv2.BORDER_CONSTANT, value=0, p=p, **kwargs["affine_args"]
        ),
        "invert": A.InvertImg(p=p),
        "gray": A.ToGray(p=p),
    }

    composed_transforms = []

    # mandatorily  resize all input images to 256x256
    composed_transforms.append(transform_dict["resize256"])


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


    composed_transforms.extend(
        [
            #A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # print(composed_transforms)

    tranform_dict = {
        "train": A.Compose(composed_transforms),
        "valid": A.Compose(
            [
                A.Resize(height=256, width=256, p=1),
                A.CenterCrop(height=224, width=224),
                # transforms.ToTensor(),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [
                A.Resize(height=256, width=256, p=1),
                A.CenterCrop(height=224, width=224),
                # transforms.ToTensor(),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    }

    return tranform_dict


# Applying transforms to the data
def get_transform(transform_type, **kwargs):
    return compose_transform(transform_type, **kwargs)


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

    # Define the scale values
    scale_min = 7
    scale_max = 13

    # shift_limit=0.05
    # scale_limit=0.9

    # r = 30
    # rotation = f"rotation_{r}_shiftlimit_{shift_limit}_scalelimit_{scale_limit}"

    # Convert scale values to a string representation
    # scale_str = f"scale_{scale_min}-{scale_max}"
    blur = f"scale_{scale_min}-{scale_max}"

    phase = "train"

    erosion_args = {}
    dilation_args = {}
    affine_args = {}
    gaussian_args = {}
    colorjitter_args = {}

    if args.transform_type == "randomcrop224,morpho_dilation":

        scales = [[3, 3], [5, 5], [7, 7], [9, 9], [11, 11], [13, 13], [15, 15]]

        for scale in scales:

            dilation_args = {}
            dilation_args["scale"] = scale

            transforms_dict = get_transform(
                transform_name,
                gaussian_args=gaussian_args,
                dilation_args=dilation_args,
                erosion_args=erosion_args,
                affine_args=affine_args,
                colorjitter_args=colorjitter_args,
            )
            print(transforms_dict["train"])

            dataloaders, class_names = get_dataloader(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                shuffle_train=False,
                add_manual_seed=True,
            )

            output_dir = os.path.join(
                "/home/vedasri/Baseline_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"scale={scale}",
            )

            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = inputs["image"]

                print(torch.mean(images))
                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, output_dir
                )
                if index == 15:
                    break

    elif args.transform_type == "randomcrop224,morpho_erosion":

        scales = [[3, 3], [5, 5], [7, 7], [9, 9], [11, 11], [13, 13], [15, 15]]

        for scale in scales:

            erosion_args = {}
            erosion_args["scale"] = scale

            transforms_dict = get_transform(
                transform_name,
                gaussian_args=gaussian_args,
                dilation_args=dilation_args,
                erosion_args=erosion_args,
                affine_args=affine_args,
                colorjitter_args=colorjitter_args,
            )
            print(transforms_dict["train"])

            dataloaders, class_names = get_dataloader(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                shuffle_train=False,
                add_manual_seed=True,
            )

            output_dir = os.path.join(
                "/home/vedasri/Baseline_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"scale={scale}",
            )

            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = inputs["image"]

                print(torch.mean(images))
                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, output_dir
                )
                if index == 15:
                    break
    
    elif args.transform_type == "randomcrop224,morpho_erosion,morpho_dilation":

        scales_erosion = [[3, 3], [5, 5], [7, 7], [9, 9], [11, 11], [13, 13], [15, 15]]
        scales_dilation = [[3, 3], [5, 5], [7, 7], [9, 9], [11, 11], [13, 13], [15, 15]]

        for scale1 in scales_erosion:
            for scale2 in scales_dilation:

                dilation_args = {}
                dilation_args["scale"] = scale1
                erosion_args["scale"] = scale2

                transforms_dict = get_transform(
                    transform_name,
                    gaussian_args=gaussian_args,
                    dilation_args=dilation_args,
                    erosion_args=erosion_args,
                    affine_args=affine_args,
                    colorjitter_args=colorjitter_args,
                )
                print(transforms_dict["train"])

                dataloaders, class_names = get_dataloader(
                    train_directory=train_directory,
                    transforms_dict=transforms_dict,
                    args=args,
                    logger=logger,
                    shuffle_train=False,
                    add_manual_seed=True,
                )

                output_dir = os.path.join(
                    "/home/vedasri/Baseline_V2/augmentation_visuals",
                    phase,
                    args.transform_type,
                    f"scale_erosion={scale1}_scale_dilation={scale2}",
            
                )

                for index, (inputs, labels) in enumerate(dataloaders[phase]):
                    # print(labels)

                    images = inputs["image"]

                    print(torch.mean(images))
                    save_image_batch(
                        images, f"{index}_" + transform_name + "_" + phase, output_dir
                    )
                    if index == 15:
                        break




    elif (
        args.transform_type == "randomcrop224"
        or args.transform_type == "resize256"
        or args.transform_type == "randomcrop224,hflip"
        or args.transform_type == "randomcrop224,invert"
        or args.transform_type == "randomcrop224,gray"
    ):

        transforms_dict = get_transform(
            transform_name,
            gaussian_args=gaussian_args,
            dilation_args=dilation_args,
            erosion_args=erosion_args,
            affine_args=affine_args,
            colorjitter_args=colorjitter_args,
        )
        print(transforms_dict["train"])

        dataloaders, class_names = get_dataloader(
            train_directory=train_directory,
            transforms_dict=transforms_dict,
            args=args,
            logger=logger,
            shuffle_train=False,
            add_manual_seed=True,
        )

        output_dir = os.path.join(
            "/home/vedasri/Baseline_V2/augmentation_visuals",
            phase,
            args.transform_type,
        )

        for index, (inputs, labels) in enumerate(dataloaders[phase]):
            # print(labels)

            images = inputs["image"]

            print(torch.mean(images))
            save_image_batch(
                images, f"{index}_" + transform_name + "_" + phase, output_dir
            )
            if index == 15:
                break

    elif args.transform_type == "randomcrop224,gaussianblur":

        blur_limits = [
            [3, 3],
            [5, 5],
            [7, 7],
            [9, 9],
            [11, 11],
            [13, 13],
            [15, 15],
            [29, 29],
            [33, 33],
        ]

        for blur_limit in blur_limits:

            gaussian_args = {}
            gaussian_args["blur_limit"] = blur_limit

            transforms_dict = get_transform(
                transform_name,
                dilation_args=dilation_args,
                gaussian_args=gaussian_args,
                erosion_args=erosion_args,
                affine_args=affine_args,
                colorjitter_args=colorjitter_args,
            )
            print(transforms_dict["train"])

            dataloaders, class_names = get_dataloader(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                shuffle_train=False,
                add_manual_seed=True,
            )

            output_dir = os.path.join(
                "/home/vedasri/Baseline_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"blur_limit={blur_limit}",
            )

            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = inputs["image"]

                print(torch.mean(images))
                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, output_dir
                )
                if index == 15:
                    break

    elif args.transform_type == "randomcrop224,affine":

        shift_limits = [0.05, 0.1, 0.15, 0.2]
        scale_limits = [0.05, 0.1, 0.15, 0.2]
        rotate_limits = [5, 10, 15, 20, 30, 40]

        for shift_limit in shift_limits:
            for scale_limit in scale_limits:
                for rotate_limit in rotate_limits:

                    affine_args = {}
                    affine_args["shift_limit"] = shift_limit
                    affine_args["scale_limit"] = scale_limit
                    affine_args["rotate_limit"] = rotate_limit

                    transforms_dict = get_transform(
                        transform_name,
                        gaussian_args=gaussian_args,
                        dilation_args=dilation_args,
                        erosion_args=erosion_args,
                        affine_args=affine_args,
                        colorjitter_args=colorjitter_args,
                    )
                    print(transforms_dict["train"])

                    dataloaders, class_names = get_dataloader(
                        train_directory=train_directory,
                        transforms_dict=transforms_dict,
                        args=args,
                        logger=logger,
                        shuffle_train=False,
                        add_manual_seed=True,
                    )

                    output_dir = os.path.join(
                        "/home/vedasri/Baseline_V2/augmentation_visuals",
                        phase,
                        args.transform_type,
                        f"shift_limit={shift_limit}_scale_limit={scale_limit}_rotate_limit={rotate_limit}",
                    )

                    for index, (inputs, labels) in enumerate(dataloaders[phase]):
                        # print(labels)

                        images = inputs["image"]

                        print(torch.mean(images))
                        save_image_batch(
                            images,
                            f"{index}_" + transform_name + "_" + phase,
                            output_dir,
                        )
                        if index == 15:
                            break

    elif args.transform_type == "randomcrop224,colorjitter":

        brightness_limits = [0.8, 0.9, 1]
        contrast_limits = [0.8, 0.9, 1]
        saturation_limits = [0.8, 0.9, 1]
        hue_limits = [-0.5, -0.2, 0, 0.2, 0.5]

        # brightness=(0.8, 1),
        # contrast=(0.8, 1),
        # saturation=(0.8, 1),
        # hue=(-0.5, 0.5),

        for brightness_limit in brightness_limits:
            for contrast_limit in contrast_limits:
                for saturation_limit in saturation_limits:
                    for hue_limit in hue_limits:

                        colorjitter_args = {}
                        colorjitter_args["brightness_limit"] = brightness_limit
                        colorjitter_args["contrast_limit"] = contrast_limit
                        colorjitter_args["saturation_limit"] = saturation_limit
                        colorjitter_args["hue_limit"] = hue_limit

                        transforms_dict = get_transform(
                            transform_name,
                            gaussian_args=gaussian_args,
                            dilation_args=dilation_args,
                            erosion_args=erosion_args,
                            affine_args=affine_args,
                            colorjitter_args=colorjitter_args,
                        )
                        print(transforms_dict["train"])

                        dataloaders, class_names = get_dataloader(
                            train_directory=train_directory,
                            transforms_dict=transforms_dict,
                            args=args,
                            logger=logger,
                            shuffle_train=False,
                            add_manual_seed=True,
                        )

                        output_dir = os.path.join(
                            "/home/vedasri/Baseline_V2/augmentation_visuals",
                            phase,
                            args.transform_type,
                            f"brightness_limit={brightness_limit}_contrast_limit={contrast_limit}_saturation_limit={saturation_limit}_hue_limit={hue_limit}",
                        )

                        for index, (inputs, labels) in enumerate(dataloaders[phase]):
                            # print(labels)

                            images = inputs["image"]

                            print(torch.mean(images))
                            save_image_batch(
                                images,
                                f"{index}_" + transform_name + "_" + phase,
                                output_dir,
                            )
                            if index == 15:
                                break

    else:

        raise NotImplementedError


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
    ap.add_argument("--batch_size", help="batch size", default=16, type=int)
    ap.add_argument("--loss_fn", help="loss function", default="ce", type=str)

    # args = vars(ap.parse_args())
    args = ap.parse_args()

    main(args)
