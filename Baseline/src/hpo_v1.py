import logging
import numpy as np
import torch
import json
import torchvision
import torch.optim as optim
import sys
import time
import os
import copy
import argparse
from datetime import datetime
import shutil

from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torchsummary import summary
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from src.core.logging_utils import (
    update_logger,
    save_code,
    export_conda_environment,
    set_seed,
    prepare_logger,
    write_json_lines,
    save_launch_scripts
)
from src.core.datatransform_builder_v2 import get_transform
from src.core.dataloader_builder import get_dataloader, get_dataloader_v2
from src.core.model_builder import get_model
from src.core.data_builder import get_dataset
from src.core.train_fn_ce import train_model_ce
from src.core.train_fn_triplet import train_model_triplet
from ray import train, tune
from ray.train import Checkpoint
from ray.air import session
from ray import air
from ray.tune import CLIReporter
from os.path import basename, normpath
from ray.tune import ExperimentAnalysis
from json import loads
import torchvision.transforms as transforms

base_dir = os.path.dirname(os.path.abspath(__file__))


def runner(config, args):

    for key, value in config.items():
        setattr(args, key, value)

    set_seed(args.seed)
    output_dir = session.get_trial_dir()
    cmd_argument = f"python {' '.join(sys.argv)}"
    save_code(base_dir=os.path.join(base_dir, "../"), output_dir=output_dir)
    export_conda_environment(output_dir=output_dir)

    logger = prepare_logger(args, cmd_argument, output_dir)
    logger.info("Arguments:")

    for arg in vars(args):
        logger.info(f"{arg:<20}: {getattr(args, arg)}")


    logger.info("Loading dataset ==========>")
    train_directory = get_dataset(args)
    num_classes = args.num_classes
    logger.info("Loaded dataset !")

    # Load data from folders
    transforms_dict = get_transform(args.transform_type)
    logger.info(f"transform type used is: {transforms_dict}")

    # Convert the dictionary to a JSON-compatible format
    def transform_to_str(transform):
        if isinstance(transform, transforms.Compose):
            return [str(t) for t in transform.transforms]
        return str(transform)

    json_compatible_dict = {k: transform_to_str(v) for k, v in transforms_dict.items()}


    with open(os.path.join(output_dir, "data_transform.json"), "w") as f:
        json.dump(json_compatible_dict, f, indent=4)

    dataloaders, class_names = get_dataloader_v2(
        train_directory=train_directory,
        transforms_dict=transforms_dict,
        args=args,
        logger=logger,
        num_classes=num_classes,
    )

    device = torch.device("cuda")
    model_ft = get_model(args, num_classes, logger=logger)
    model_ft = model_ft.to(device)

    logger.info("Model Summary:-\n")
    for num, (name, param) in enumerate(model_ft.named_parameters()):
        logger.info(
            f"Index: {num:3d} Layer name: {name:40s} requires grad: {param.requires_grad}, mean: {param.mean()}"
        )
    summary(model_ft, input_size=(3, 224, 224))

    criterion = torch.nn.CrossEntropyLoss()

    trainable_params = []
    trainable_params_names = []

    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_params_names.append(name)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model_ft.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    percentage_trainable = (trainable_params_count / total_params) * 100
    logger.info(f"Percentage of trainable parameters: {percentage_trainable:.2f}%")    
    logger.info(f"Trainable params: {trainable_params_names}")



    optimizer_ft = optim.Adam(trainable_params, lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    logger.info(f"Optimizer settings: {optimizer_ft}")
    logger.info(f"exp_lr_scheduler Step Size: {exp_lr_scheduler.step_size}")
    logger.info(f"exp_lr_scheduler Gamma: {exp_lr_scheduler.gamma}")


    if args.loss_fn == "ce" or args.loss_fn == "triplet_finetune_with_ce"  or args.loss_fn == "simclr_finetune_with_ce" :
        train_model_fn = train_model_ce
    elif args.loss_fn == "triplet":
        train_model_fn = train_model_triplet
    else:
        raise NotImplementedError()



    model_ft = train_model_fn(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=args.epochs,
        num_classes=num_classes,
        class_names=class_names,
        dataloaders=dataloaders,
        device=device,
        SummaryWriter=SummaryWriter,
        output_dir=output_dir,
        logger=logger,
        args= args
    )
   


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

   


    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", required=True, help="Training mode: finetue/transfer/scratch"
    )
    ap.add_argument("--arch", required=True, help="Architecture", type=str)
    ap.add_argument("--batch_size", help="batch size", default=64, type=int)
    ap.add_argument("--epochs", help="epochs", default=50, type=int)
    ap.add_argument("--transform_type", help="transform type", type=str)
    ap.add_argument("--dataset", help="dataset", type=str)
    ap.add_argument("--dropout_prob", help="dropout_prob", type=float, default=0.0)

    ap.add_argument("--hpo_search_space_config", help="dataset", type=str)
    ap.add_argument(
        "--gpu_per_worker", help="number of gpu per trial", default=1.0, type=float
    )
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--hpo_exp_name", help="exp_name", default="demo", type=str)
    ap.add_argument("--log_filepath", help="log filename", type=str)
    ap.add_argument("--hpo_restore", help="restore experiment",  type=str2bool, nargs='?', const=True, default=False)
    ap.add_argument("--loss_fn", help="loss function", default="ce", type=str)
    ap.add_argument("--triplet_embedding_size", help="embeeding size", default=64, type=int)
    ap.add_argument("--ckpt_path_dict", help="config to ckpt path dict", type=str)
    ap.add_argument("--freeze_backbone", help="Freeze backbone", type=str2bool, nargs='?', const=True, default=False)
    ap.add_argument("--num_classes", help="number os classed", default=25,
                     type=int)
    ap.add_argument("--launch_script_path", help="script used to launch this code", type=str)
    ap.add_argument("--use_imagenet_pretrained_weights", help="backbone init with imagenet weights", type=str2bool, nargs='?', const=True, default=False)

    # args = vars(ap.parse_args())
    args = ap.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(base_dir, "../results_hpo")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    os.environ["TUNE_RESULT_DIR"] = results_dir
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # exp_name = f"{args.hpo_exp_name}_{time_str}"

    exp_name = f"{args.hpo_exp_name}"

    from hpo.search_space_utils import parse_search_space

    search_space = parse_search_space(args.hpo_search_space_config)
    
    # Limit the number of rows.
    reporter = CLIReporter()
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter.add_metric_column("test_loss")
    reporter.add_metric_column("test_acc")

    exp_command = f"python {' '.join(sys.argv)}"
    save_launch_scripts(args, os.path.join(results_dir, exp_name), exp_command)

    
    if args.hpo_restore:
        trainable = tune.with_resources(
            tune.with_parameters(runner, args=args),
            resources={"cpu": 2, "gpu": args.gpu_per_worker},
        )

        tuner = tune.Tuner.restore(
            path=os.path.join(results_dir, exp_name),
            trainable=trainable,
            resume_errored=True,
        )
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

    print("===================== Starting the HPO ===========================================")

    results = tuner.fit()

    analyse_run_exp(args, results)

    if True:

        curdir = os.path.dirname(os.path.abspath(__file__))
        dst_dir = os.path.join(curdir, "../", results_dir, exp_name)
        source_path = os.path.join(curdir, "../logs", args.log_filepath)
        shutil.copy2(source_path, dst_dir)
