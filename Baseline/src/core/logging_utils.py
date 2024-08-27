import logging
import os
import re
import numpy as np
import zipfile
import json
import torch
import torchvision
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629"""

    def __init__(self, fmt):
        super().__init__()
        grey = "\x1b[38;21m"
        blue = "\x1b[38;5;39m"
        yellow = "\x1b[33;20m"
        red = "\x1b[38;5;196m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: blue + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LoggerPrecisionFilter(logging.Filter):
    def __init__(self, precision):
        super().__init__()
        self.print_precision = precision

    def str_round(self, match_res):
        return str(round(eval(match_res.group()), self.print_precision))

    def filter(self, record):
        # use regex to find float numbers and round them to specified precision
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)
        if record.msg != "":
            if re.search(r"([-+]?\d+\.\d+)", record.msg):
                record.msg = re.sub(r"([-+]?\d+\.\d+)", self.str_round, record.msg)
        return True


def update_logger(cfg, clear_before_add=False, output_dir=None):
    root_logger = logging.getLogger("Baseline")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        handler.setFormatter(CustomFormatter(fmt))

        root_logger.addHandler(handler)

    logging_level = logging.INFO
    root_logger.setLevel(logging_level)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(output_dir, "exp_print.log"))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )
    fh.setFormatter(logger_formatter)
    #root_logger.addHandler(fh)

    print_decimal_digits = 6

    # set print precision for terse logging
    np.set_printoptions(precision=print_decimal_digits)
    precision_filter = LoggerPrecisionFilter(print_decimal_digits)
    # for handler in root_logger.handlers:
    #     handler.addFilter(precision_filter)

    root_logger.propagate = True
    return root_logger

    # root_logger.info("HERE")


def save_code(base_dir, output_dir):

    if not os.path.exists(os.path.join(output_dir, "code")):
        os.makedirs(os.path.join(output_dir, "code"))

    zip_directory(
        os.path.join(base_dir, "src"), os.path.join(output_dir, "code/src.zip")
    )


def zip_directory(directory, zip_file):
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                archname = os.path.relpath(file_path, directory)
                zipf.write(file_path, archname)


import subprocess


def export_conda_environment(output_dir, filename="environment.yml"):
    try:
        # Execute the command
        subprocess.run(
            f"conda env export > {os.path.join(output_dir, filename)}",
            shell=True,
            check=True,
        )
        print(f"Conda environment exported to {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting conda environment: {e}")





def write_json_lines(file_path, list_of_dicts):
    with open(file_path, "w") as json_file:
        for dictionary in list_of_dicts:
            json.dump(dictionary, json_file)
            json_file.write("\n")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_logger(args,cmd_argument, output_dir):
    logger = update_logger(args, clear_before_add=False, output_dir=output_dir)
    logger.info(f"Run command: {cmd_argument}")
    command_file = open(os.path.join(output_dir, "run_commond.log"), "w")
    command_file.write(cmd_argument)
    command_file.close()
    logger.info(f"NumPy version         : {np.__version__}")
    logger.info(f"PyTorch version       : {torch.__version__}")
    logger.info(f"Torchvision version   : {torchvision.__version__}")
    logger.info(f"Python version        : {sys.version}")

    return logger


import os
import shutil
from datetime import datetime


def remove_argument(command, arg_name):
    import shlex
    
    # Split the command into parts
    parts = shlex.split(command)
    
    # Find the argument and remove it along with its value
    if arg_name in parts:
        index = parts.index(arg_name)
        if index + 1 < len(parts):
            parts.pop(index + 1)  # Remove the value
        parts.pop(index)  # Remove the argument itself
            
    # Reconstruct the command
    return ' '.join(parts)


def save_launch_scripts(args, results_dir, exp_command):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Corrected the strftime format for seconds

    script_filename = os.path.basename(args.launch_script_path)
    script_filename = os.path.splitext(script_filename)[0]

    scripts_dir = os.path.join(results_dir, "scripts")
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)

    dest_path = os.path.join(scripts_dir, f"{script_filename}_{timestamp}.sh")

    if os.path.exists(args.launch_script_path):
        shutil.copy(args.launch_script_path, dest_path)

    ss_filename = os.path.basename(args.hpo_search_space_config)
    ss_filename = os.path.splitext(ss_filename)[0]

    ss_saved_path = os.path.join(scripts_dir, f"{ss_filename}_{timestamp}.yaml")
    shutil.copy(args.hpo_search_space_config, ss_saved_path)

    # Corrected parentheses for 'timestamp'
    with open(os.path.join(scripts_dir, f"run_command_{timestamp}.log"), 'w') as f:
        f.write("{}".format(exp_command))

    device_ids = os.getenv("CUDA_VISIBLE_DEVICES")

    modified_exp_command = remove_argument(exp_command, "--hpo_search_space_config")
    restore_argument = f"CUDA_VISIBLE_DEVICES={device_ids} " + modified_exp_command + " --hpo_restore True " + f" --hpo_search_space_config {ss_saved_path}"
    # Corrected parentheses for 'timestamp'
    with open(os.path.join(scripts_dir, f"restore_command_{timestamp}.log"), 'w') as f:
        f.write("{}".format(restore_argument))

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Corrected and completed the zip_directory function call
    zip_directory(os.path.join(base_dir, "../", "src"), os.path.join(results_dir, "scripts", "src.zip"))
    zip_directory(os.path.join(base_dir, "../", "scripts"), os.path.join(results_dir, "scripts", "scripts.zip"))

 