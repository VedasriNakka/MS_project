"""

python datatransform_builder_v2.py --transform_type resize256,affine
python ./src/data_aug/datatransform_visuals.py --transform_type randomcrop198,morpho_dilation

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
from src.data_aug.dataloader_builder import get_dataloader_v2
from models.resnet_simclr import ResNetSimCLR
from src.simclr import SimCLR
import torchvision


import torchvision.transforms as transforms


def compose_transform(transform_types, **kwargs):

    p=1
    transform_dict = {
        "default": A.Resize(height=256, width=256, p=1.0),
        "resize256": A.Resize(height=256, width=256, p=1),
        "randomcrop224": A.RandomCrop(224, 224, p=1.0),    #77%
        "randomcrop198": A.RandomCrop(198, 198, p=1.0),    #60% 
        "randomcrop128": A.RandomCrop(128, 128, p=1.0),    #28%
        "randomcrop180": A.RandomCrop(180, 180, p=1.0),    #50%
        "randomcrop161": A.RandomCrop(161, 161, p=1.0),    #40%
        "randomcrop214": A.RandomCrop(214, 214, p=1.0),    #70% 
        "randomcrop228": A.RandomCrop(228, 228, p=1.0),    #80%
        "randomcrop256": A.RandomCrop(256, 256, p=1.0),

        # scale hyperparam
        "morpho_dilation": A.Morphological(scale=(7, 7), operation="dilation", p=p), **kwargs["dilation_args"],
        # scale hyperparam
        "morpho_erosion": A.Morphological(scale=(7, 7), operation="erosion", p=p), **kwargs["dilation_args"],
        "hflip": A.HorizontalFlip(p=p),
        "colorjitter": A.ColorJitter(
            brightness=(0.8, 1),
            contrast=(0.8, 1),
            saturation=(0.8, 1),
            hue=(-0.5, 0.5),
            always_apply=False,
            p=p,
            **kwargs["dilation_args"],
        ),
        # blur_limit
        "gaussianblur": A.GaussianBlur(
            blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=p, **kwargs["dilation_args"],
        ),
        #  shift, scale, rotate
        "affine": A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=p,
            **kwargs["dilation_args"],
        ),
        "invert": A.InvertImg(p=p),
        "gray": A.ToGray(p=p),
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
def get_transform(transform_type, **kwargs):
    return compose_transform(transform_type, **kwargs)


def get_dataset(args):
    if args.dataset == "icdar":
        train_directory = "/home/vedasri/datasets/HomerCompTraining_Cropped"

    elif args.dataset == "alpub":
        train_directory = "/home/vedasri/datasets/alpub_v2/images"
    else:
        raise NotImplementedError

    return train_directory

def save_image_batch(batch_data, phase, index, log_dir):
    # IMG_MEAN = [0.0, 0., 0.]
    # IMG_STD = [1.0, 1.0, 1.0]

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        logger.info(f"ten shape: {ten.shape}")
        logger.info(f"ten size: {ten.size}")
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    #if(args.epochs <= 5):
    # tensor_x: size [B, 3, H, W]
    torchvision.utils.save_image(
        denormalize(batch_data),
        os.path.join(log_dir, f"batch_data_{phase}_{index}.png"),
    )
    logger.info("saved batch data in the image format to the output folder!")


def main(args):

    train_directory = get_dataset(args)

    transform_name = args.transform_type

    phase = "train"

    erosion_args = {}
    dilation_args = {}
    affine_args = {}
    gaussian_args = {}
    colorjitter_args = {}

    if args.transform_type == "randomcrop228,morpho_dilation":

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

            dataloaders, class_names = get_dataloader_v2(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                num_classes=args.num_classes,
            )

            output_dir = os.path.join(
                "/home/vedasri/SimCLR_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"scale={scale}",
            )

            os.makedirs(output_dir, exist_ok=True)

            for index, (images, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = [img["image"] for img in images]
                images = torch.cat(images, dim=0)
                images = images.to(args.device)

                print(torch.mean(images))

                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, index, output_dir
                )
                #print("--------------Images are saving----------------")
                if index == 5:
                    break

    
    elif args.transform_type == "randomcrop228,colorjitter":

        brightness_limits = [0.8, 0.9, 1]
        contrast_limits = [0.8, 0.9, 1]
        saturation_limits = [0.8, 0.9, 1]
        hue_limits = [-0.5, -0.2, 0, 0.2, 0.5]


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

                        dataloaders, class_names = get_dataloader_v2(
                            train_directory=train_directory,
                            transforms_dict=transforms_dict,
                            args=args,
                            logger=logger,
                            num_classes=args.num_classes,
                        )

                        output_dir = os.path.join(
                            "/home/vedasri/SimCLR_V2/augmentation_visuals",
                            phase,
                            args.transform_type,
                            f"brightness_limit={brightness_limit}_contrast_limit={contrast_limit}_saturation_limit={saturation_limit}_hue_limit={hue_limit}",
                        )

                        os.makedirs(output_dir, exist_ok=True)

                        for index, (images, labels) in enumerate(dataloaders[phase]):
                            # print(labels)

                            images = [img["image"] for img in images]
                            images = torch.cat(images, dim=0)
                            images = images.to(args.device)

                            print(torch.mean(images))

                            save_image_batch(
                                images, f"{index}_" + transform_name + "_" + phase, index, output_dir
                            )
                            #print("--------------Images are saving----------------")
                            if index == 5:
                                break

    elif args.transform_type == "randomcrop198,colorjitter,hflip":

        brightness_limits = [0.8, 0.9, 1]
        contrast_limits = [0.8, 0.9, 1]
        saturation_limits = [0.8, 0.9, 1]
        hue_limits = [-0.5, -0.2, 0, 0.2, 0.5]


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

                        dataloaders, class_names = get_dataloader_v2(
                            train_directory=train_directory,
                            transforms_dict=transforms_dict,
                            args=args,
                            logger=logger,
                            num_classes=args.num_classes,
                        )

                        output_dir = os.path.join(
                            "/home/vedasri/SimCLR_V2/augmentation_visuals",
                            phase,
                            args.transform_type,
                            f"brightness_limit={brightness_limit}_contrast_limit={contrast_limit}_saturation_limit={saturation_limit}_hue_limit={hue_limit}",
                        )

                        os.makedirs(output_dir, exist_ok=True)

                        for index, (images, labels) in enumerate(dataloaders[phase]):
                            # print(labels)

                            images = [img["image"] for img in images]
                            images = torch.cat(images, dim=0)
                            images = images.to(args.device)

                            print(torch.mean(images))

                            save_image_batch(
                                images, f"{index}_" + transform_name + "_" + phase, index, output_dir
                            )
                            #print("--------------Images are saving----------------")
                            if index == 5:
                                break


    elif (args.transform_type == "randomcrop198" 
        or args.transform_type == "randomcrop198,hflip"
        or args.transform_type == "randomcrop198,invert"
        or args.transform_type == "randomcrop198,gray"):

        transforms_dict = get_transform(
                transform_name,
                gaussian_args=gaussian_args,
                dilation_args=dilation_args,
                erosion_args=erosion_args,
                affine_args=affine_args,
                colorjitter_args=colorjitter_args,
            )
        print(transforms_dict["train"])

        dataloaders, class_names = get_dataloader_v2(
            train_directory=train_directory,
            transforms_dict=transforms_dict,
            args=args,
            logger=logger,
            num_classes=args.num_classes,
        )

        output_dir = os.path.join(
            "/home/vedasri/SimCLR_V2/augmentation_visuals",
            phase,
            args.transform_type,
        )

        os.makedirs(output_dir, exist_ok=True)

        for index, (images, labels) in enumerate(dataloaders[phase]):
            # print(labels)

            images = [img["image"] for img in images]
            images = torch.cat(images, dim=0)
            images = images.to(args.device)

            print(torch.mean(images))

            save_image_batch(
                images, f"{index}_" + transform_name + "_" + phase, index, output_dir
            )
            #print("--------------Images are saving----------------")
            if index == 5:
                break

    elif args.transform_type == "randomcrop198,morpho_erosion,morpho_dilation":

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

            dataloaders, class_names = get_dataloader_v2(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                num_classes=args.num_classes,
            )

            output_dir = os.path.join(
                "/home/vedasri/SimCLR_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"scale={scale}",
            )

            os.makedirs(output_dir, exist_ok=True)

            for index, (images, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = [img["image"] for img in images]
                images = torch.cat(images, dim=0)
                images = images.to(args.device)

                print(torch.mean(images))

                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, index, output_dir
                )
                print("--------------Images are saving----------------")
                if index == 5:
                    break

    elif args.transform_type == "randomcrop198,morpho_dilation,morpho_erosion":

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

            dataloaders, class_names = get_dataloader_v2(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                num_classes=args.num_classes,
            )

            output_dir = os.path.join(
                "/home/vedasri/SimCLR_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"scale={scale}",
            )

            os.makedirs(output_dir, exist_ok=True)

            for index, (images, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = [img["image"] for img in images]
                images = torch.cat(images, dim=0)
                images = images.to(args.device)

                print(torch.mean(images))

                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, index, output_dir
                )
                print("--------------Images are saving----------------")
                if index == 5:
                    break

    elif args.transform_type == "randomcrop198,gaussianblur":

        scales = [[3, 3],
            [5, 5],
            [7, 7],
            [9, 9],
            [11, 11],
            [13, 13],
            [15, 15],
            [29, 29],
            [33, 33]],

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

            dataloaders, class_names = get_dataloader_v2(
                train_directory=train_directory,
                transforms_dict=transforms_dict,
                args=args,
                logger=logger,
                num_classes=args.num_classes,
            )

            output_dir = os.path.join(
                "/home/vedasri/SimCLR_V2/augmentation_visuals",
                phase,
                args.transform_type,
                f"scale={scale}",
            )

            os.makedirs(output_dir, exist_ok=True)

            for index, (images, labels) in enumerate(dataloaders[phase]):
                # print(labels)

                images = [img["image"] for img in images]
                images = torch.cat(images, dim=0)
                images = images.to(args.device)

                print(torch.mean(images))

                save_image_batch(
                    images, f"{index}_" + transform_name + "_" + phase, index, output_dir
                )
                print("--------------Images are saving----------------")
                if index == 5:
                    break


    else:
        raise NotImplementedError

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', "True", 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', "False", 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    ap.add_argument("--batch_size", help="batch size", default=8, type=int)
    ap.add_argument("--arch", help="model architecture", default="resnet18", type=str)
    ap.add_argument("--out_dim", help="output dimension", default=128, type=int)
    ap.add_argument("--lr", help="learning rate", default=0.001, type=float)
    ap.add_argument("--weight_decay", help="weight decay", default=1e-6, type=float)
    ap.add_argument("--loss_fn", help="loss function", default="info_nce", type=str)
    ap.add_argument("--num_classes", help="number of classes", default=25, type=int)
    ap.add_argument("--n_views", help="number of views", default=2, type=int)
    ap.add_argument("--use_imagenet_pretrained_weights", 
                        help="backbone init with imagenet weights", 
                        type=str2bool, 
                        nargs='?', 
                        const=True, 
                        default=False)
    ap.add_argument(
        "--fp16-precision",
        action="store_true",
        help="Whether or not to use 16-bit precision GPU training.",
    )
    ap.add_argument(
        "--epochs",
        default=1,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    ap.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="softmax temperature (default: 0.07)",
    )
    
    args = ap.parse_args()

    args.device = torch.device(f"cuda")
    #args.n_views == 2



    main(args)
