import argparse
import torch
import os
import sys

from datetime import datetime
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import logging
from torch.utils.data import random_split
import random
import numpy as np


def main(args):

    random.seed(args.seed)  # python random generator
    np.random.seed(args.seed)  # numpy random generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if True:

        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)

        # Get the directory of the script (Baseline/src/train.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the logs directory in the parent directory (Baseline/logs)

        # Generate filename based on current date and time
        current_datetime = datetime.now().strftime("%m_%d_%H-%M-%S")

        # creating folder with current run and checkpoint
        results_dir = os.path.join(os.path.dirname(script_dir), "results")
        os.makedirs(results_dir, exist_ok=True)

        output_dir = os.path.join(
            results_dir,
            f"output_arch={args.arch}_epochs={args.epochs}_transform_type={args.transform_type}_{current_datetime}",
        )
        os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"training.log")

        # Configure logging for the second folder
        logger = logging.getLogger()  # "second_logger")
        logger.setLevel(logging.INFO)
        logger_handler = logging.FileHandler(output_file)
        logger_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(logger_handler)

        # Create a stream handler and set its level to INFO
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        # Set the formatter for the stream handler
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Add the stream handler to the logger
        logger.addHandler(stream_handler)

        # logger.addHandler(formatter)

        # logger.propagate = False

    else:

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Get the directory of the script (Baseline/src/train.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the logs directory in the parent directory (Baseline/logs)

        # Generate filename based on current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # creating folder with current run and checkpoint
        results_dir = os.path.join(os.path.dirname(script_dir), "results")
        os.makedirs(results_dir, exist_ok=True)

        output_dir = os.path.join(results_dir, f"output_{current_datetime}")
        os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"training.log")
        logger_handler = logging.FileHandler(output_file)
        logger_handler = logging.FileHandler(output_file)
        logger_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(logger_handler)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg:<20}: {getattr(args, arg)}")

    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_index}")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(root_folder=args.data, args=args)

    train_dataset = dataset.get_dataset(
        name=args.dataset_name, n_views=args.n_views, transform_type=args.transform_type
    )
    logger.info(f"Length of full dataset: {len(train_dataset)}")
    logger.info(f"transform type used is: {args.transform_type}")

    # Define the sizes of each split (e.g., 70% train, 15% validation, 15% test)
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    # Use random_split to split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size]
    )

    logger.info(f"Training dataset size    : {len(train_dataset)}")
    logger.info(f"Validation dataset size  : {len(val_dataset)}")
    logger.info(f"Test dataset size        : {len(test_dataset)}")

    # exit()

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # Create iterators for data loading
    dataloaders = {
        "train_loader": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        ),  # dataset['train']
        "valid_loader": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        ),
        "test_loader": torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        ),
    }

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    # logger.info(f"Model device: {model.backbone.device}")

    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    train_loader = dataloaders["train_loader"]
    valid_loader = dataloaders["valid_loader"]
    test_loader = dataloaders["test_loader"]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    # with torch.cuda.device(args.gpu_index):

    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for phase in ["train", "valid", "test"]:

        simclr = SimCLR(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            output_dir=output_dir,
        )
        simclr.train(train_loader, valid_loader, test_loader)


if __name__ == "__main__":

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="PyTorch SimCLR")
    parser.add_argument(
        "-data", metavar="DIR", default="../vedasri/datasets", help="path to dataset"
    )
    parser.add_argument(
        "--dataset-name",
        default="HomerCompTraining_Cropped",
        help="dataset name",
        choices=[
            "stl10",
            "cifar10",
            "Cropped_Images_New",
            "HomerCompTraining_Cropped",
            "icdar",
        ],
    )

    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=12,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.0003,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--fp16-precision",
        action="store_true",
        help="Whether or not to use 16-bit precision GPU training.",
    )

    parser.add_argument(
        "--out_dim", default=128, type=int, help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--log-every-n-steps", default=100, type=int, help="Log every n steps"
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="softmax temperature (default: 0.07)",
    )
    parser.add_argument(
        "--n-views",
        default=2,
        type=int,
        metavar="N",
        help="Number of views for contrastive learning training.",
    )
    parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")
    parser.add_argument("--transform_type", help="transform type", type=str)

    args = parser.parse_args()

    main(args)
