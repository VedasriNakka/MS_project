import argparse
import torch
import os
import sys
import json
import shutil
import time

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
from sklearn.metrics import confusion_matrix
from src.core.visualization_utils import draw_confusion_matrix, save_image_batch

from hpo.search_space_utils import parse_search_space
from src.data_builder import get_dataset_name

from ray import train, tune
from ray.train import Checkpoint
from ray.air import session
from ray import air
from ray.tune import CLIReporter
from os.path import basename, normpath
from ray.tune import ExperimentAnalysis
from json import loads
from src.hpo.logging_utils import update_logger, save_code, export_conda_environment

base_dir = os.path.dirname(os.path.abspath(__file__))


def write_json_lines(file_path, list_of_dicts):
    with open(file_path, "w") as json_file:
        for dictionary in list_of_dicts:
            json.dump(dictionary, json_file)
            json_file.write("\n")


def runner(config, args):

    for key, value in config.items():
        setattr(args, key, value)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # results_dir = os.path.join(os.path.dirname(script_dir), f"results_{args.dataset}", args.arch)

    output_dir = session.get_trial_dir()

    logger = update_logger(args, clear_before_add=True, output_dir=output_dir)
    save_code(base_dir=os.path.join(base_dir, "../"), output_dir=output_dir)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg:<20}: {getattr(args, arg)}")

    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    args.device = torch.device(f"cuda")

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

    simclr = SimCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        output_dir=output_dir,
    )
    simclr.train(train_loader, valid_loader, test_loader)


def analyse_run_exp(args, results):

    logdir = results.experiment_path
    metric = "valid_acc"
    mode = "max"
    analysis = ExperimentAnalysis(logdir, default_metric=metric, default_mode=mode)

    df = analysis.dataframe(metric=metric, mode=mode)
    output_json = loads(df.to_json(orient="records"))
    filepath = os.path.join(logdir, f"best_epoch_in_each_trial_by_{metric}.json")
    write_json_lines(filepath, output_json)

    if mode == "min":
        idx = df[metric].idxmin()

    else:
        idx = df[metric].idxmax()

    filepath = os.path.join(logdir, f"best_epoch_across_all_trial_by_{metric}.json")

    best_iteration_res = output_json[idx]

    for result in results:
        if result.metrics["trial_id"] == df.iloc[idx].trial_id:
            best_iteration_res["path"] = basename(normpath(result.path))
            best_iteration_res["config"] = result.config
            best_iteration_res["rank_metric"] = metric

    with open(filepath, "w") as f:
        json.dump(best_iteration_res, f, indent=4)


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
    # parser.add_argument(
    #     "--dataset_name_default",
    #     default="HomerCompTraining_Cropped",
    #     help="dataset name",
    #     choices=["stl10", "cifar10", "Cropped_Images_New", "HomerCompTraining_Cropped"],
    # )

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
        "--batch_size",
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
    parser.add_argument("--hpo_search_space_config", help="dataset", type=str)
    parser.add_argument("--hpo_exp_name", help="exp_name", default="demo", type=str)
    parser.add_argument("--hpo_restore", help="restore experiment", type=bool)
    parser.add_argument("--dataset_name", help="icdar", type=str)
    parser.add_argument("--log_filepath", help="log filename", type=str)
    parser.add_argument(
        "--gpu_per_worker", help="number of gpu per trial", default=1.0, type=float
    )

    args = parser.parse_args()

    results_dir = "/home/vedasri/SimCLR/results_hpo"
    os.makedirs(results_dir, exist_ok=True)

    os.environ["TUNE_RESULT_DIR"] = results_dir
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{args.hpo_exp_name}_{time_str}"

    search_space = parse_search_space(args.hpo_search_space_config)

    # Limit the number of rows.
    reporter = CLIReporter()
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter.add_metric_column("test_loss")
    reporter.add_metric_column("test_acc")

    if args.hpo_restore:
        trainable = tune.with_resources(
            tune.with_parameters(runner, args=args),
            resources={"cpu": 3, "gpu": args.gpu_per_worker},
        )

        tuner = tune.Tuner.restore(
            path=os.path.join(results_dir, exp_name),
            trainable=trainable,
            resume_errored=True,
        )
        print("results_dir", results_dir)

    else:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(runner, args=args),
                resources={"cpu": 2, "gpu": args.gpu_per_worker},
            ),
            tune_config=tune.TuneConfig(
                metric="valid_acc",
                mode="max",
            ),
            run_config=air.RunConfig(
                local_dir=results_dir, progress_reporter=reporter, name=exp_name
            ),
            param_space=search_space,
        )

    results = tuner.fit()

    analyse_run_exp(args, results)

    if True:

        curdir = os.path.dirname(os.path.abspath(__file__))
        dst_dir = os.path.join(curdir, "../", results_dir, exp_name)
        source_path = os.path.join(curdir, "../logs", args.log_filepath)
        shutil.copy2(source_path, dst_dir)
