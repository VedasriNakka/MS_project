import argparse
from torchvision import models



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', "True", 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', "False", 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

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

    #parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")
    parser.add_argument("--transform_type", help="transform type", type=str)
    parser.add_argument("--hpo_search_space_config", help="dataset", type=str)
    parser.add_argument("--hpo_exp_name", help="exp_name", default="demo", type=str)
    #parser.add_argument("--hpo_restore", help="restore experiment", type=bool)
    parser.add_argument("--dataset_name", help="icdar", type=str)
    parser.add_argument("--num_classes", help="number of classes", default=25, type=int)
    parser.add_argument("--log_filepath", help="log filename", type=str)
    parser.add_argument(
        "--gpu_per_worker", help="number of gpu per trial", default=1.0, type=float
    )
    parser.add_argument("--loss_fn", help="loss function", default="info_nce", type=str)
    parser.add_argument(
        "--cpu_per_worker", help="number of cpu per trial", default=2.0, type=float
    )

    parser.add_argument("--launch_script_path", help="script used to launch this code", type=str)
    parser.add_argument("--use_imagenet_pretrained_weights", 
                        help="backbone init with imagenet weights", 
                        type=str2bool, 
                        nargs='?', 
                        const=True, 
                        default=True)
    parser.add_argument("--hpo_restore", 
                        help="restore experiment", 
                        type=bool,
                        nargs='?', 
                        const=True,
                        default=False)

    args = parser.parse_args()
    return args
