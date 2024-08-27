import torch
import os
import sys
import json
import shutil

from datetime import datetime
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from hpo.search_space_utils import parse_search_space
from ray import tune
from ray.air import session
from ray import air
from ray.tune import CLIReporter
from os.path import basename, normpath
from ray.tune import ExperimentAnalysis
from json import loads
from src.hpo.logging_utils import (
    save_code,
    export_conda_environment,
    prepare_logger,
    write_json_lines,
    save_launch_scripts,
    set_seed
)
from src.data_aug.data_builder import get_dataset
from src.data_aug.dataloader_builder import get_dataloader_v2
from src.data_aug.datatransform_builder import get_transform
from src.hpo.logging_utils import set_seed
from src.parse_args import get_args

base_dir = os.path.dirname(os.path.abspath(__file__))


def runner(config, args):

    set_seed(args.seed)
    for key, value in config.items():
        setattr(args, key, value)

    output_dir = session.get_trial_dir()
    cmd_argument = f"python {' '.join(sys.argv)}"
    logger = prepare_logger(args, cmd_argument, output_dir)
    save_code(base_dir=os.path.join(base_dir, "../"), output_dir=output_dir)
    export_conda_environment(output_dir=output_dir)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg:<20}: {getattr(args, arg)}")

    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."


    args.device = torch.device(f"cuda")
    train_directory = get_dataset(args)
    num_classes = args.num_classes

    transforms_dict = get_transform(args.transform_type)
    logger.info(f"transform type used is: {transforms_dict}")

    dataloaders, class_names = get_dataloader_v2(
        train_directory=train_directory,
        transforms_dict=transforms_dict,
        args=args,
        logger=logger,
        num_classes=num_classes,
    )

    model = ResNetSimCLR(base_model=args.arch, 
                         out_dim=args.out_dim, 
                         use_pretrained=args.use_imagenet_pretrained_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        args.lr, 
        weight_decay=args.weight_decay
    )

    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]
    test_loader = dataloaders["test"]

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



def check_args(args):
    if args.dataset_name == "icdar":
        assert args.num_classes == 25, "Wrong number of classes selected. Please input 25 classes for icdar"
    elif args.dataset_name == "alpub":
        assert args.num_classes == 24, "Wrong number of classes selected. Please input 24 classes for alpub"
    else:
        raise NotImplementedError()

if __name__ == "__main__":

    args = get_args()
    check_args(args)

    results_dir = os.path.join(base_dir, "../results_hpo")
    os.makedirs(results_dir, exist_ok=True)

    os.environ["TUNE_RESULT_DIR"] = results_dir
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #exp_name = f"{args.hpo_exp_name}_{time_str}"
    exp_name = f"{args.hpo_exp_name}"
    search_space = parse_search_space(args.hpo_search_space_config)

    reporter = CLIReporter()
    reporter.add_metric_column("test_loss")
    reporter.add_metric_column("test_acc")

    exp_command = f"python {' '.join(sys.argv)}"
    save_launch_scripts(args, os.path.join(results_dir, exp_name), exp_command)

    if args.hpo_restore:
        trainable = tune.with_resources(
            tune.with_parameters(runner, args=args),
            resources={"cpu": args.cpu_per_worker, 
                       "gpu": args.gpu_per_worker},
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
                resources={"cpu": 2, 
                           "gpu": args.gpu_per_worker},
            ),
            tune_config=tune.TuneConfig(
                metric="valid_acc",
                mode="max",
            ),
            run_config=air.RunConfig(
                local_dir=results_dir, 
                progress_reporter=reporter, 
                name=exp_name
            ),
            param_space=search_space,
        )

    results = tuner.fit()
    analyse_run_exp(args, results)


    if True:
        from os.path import basename, normpath
        trial_config_2_path = {}
        for result in results:
            trial_path = basename(normpath(result.path))
            trial_config = result.config["transform_type"]
            trial_config_2_path[trial_config] =  os.path.join(result.path, 
                                                              'best_checkpoint.pth.tar')

        with open(os.path.join(results.experiment_path, 
                               "config2ckpt_path.json"),
                                "w") as f:
            json.dump(trial_config_2_path, f, indent=4)

    if True:
        curdir = os.path.dirname(os.path.abspath(__file__))
        dst_dir = os.path.join(curdir, "../", results_dir, exp_name)
        source_path = os.path.join(curdir, "../logs", args.log_filepath)
        shutil.copy2(source_path, dst_dir)
