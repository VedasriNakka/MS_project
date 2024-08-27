import logging

import numpy as np
import torch
import json
import torchvision
import torch.utils.data as data
import torch.optim as optim
import sys
import time
import os
import copy
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

# from nets import *
from torchsummary import summary
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
from src.core.logging_utils import update_logger, save_code, export_conda_environment

# from src.core.datatransform_builder import get_transform
from src.core.datatransform_builder_v2 import get_transform

from src.core.dataloader_builder import get_dataloader
from src.core.model_builder import get_model
from src.core.visualization_utils import draw_confusion_matrix, save_image_batch
from src.core.data_builder import get_dataset

base_dir = os.path.dirname(os.path.abspath(__file__))


def main(args):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(
        os.path.dirname(script_dir), f"results_{args.dataset}", args.arch
    )
    os.makedirs(results_dir, exist_ok=True)

    output_dir = os.path.join(
        results_dir,
        f"results_epochs={args.epochs}_transform_type={args.transform_type}_{current_datetime}",
    )
    os.makedirs(output_dir)

    cmd_argument = f"python {' '.join(sys.argv)}"

    save_code(base_dir=os.path.join(base_dir, "../"), output_dir=output_dir)

    if False:

        output_file = os.path.join(output_dir, f"training.log")

        # Configure logging for the second folder
        logger = logging.getLogger("logger")
        logger.setLevel(logging.INFO)
        logger_handler = logging.FileHandler(output_file)
        logger_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(logger_handler)

        # Create a stream handler and set its level to INFO
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        logger.addHandler(stream_handler)

    else:
        logger = update_logger(args, clear_before_add=True, output_dir=output_dir)

    logger.info(f"Run command: {cmd_argument}")
    export_conda_environment(output_dir=output_dir)
    command_file = open(os.path.join(output_dir, "run_commond.log"), "w")
    command_file.write(cmd_argument)
    command_file.close()

    logger.info(f"NumPy version         : {np.__version__}")
    logger.info(f"PyTorch version       : {torch.__version__}")
    logger.info(f"Torchvision version   : {torchvision.__version__}")
    logger.info(f"Python version        : {sys.version}")

    ##exit()

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg:<20}: {getattr(args, arg)}")

    # Set training mode

    train_directory = get_dataset(args)

    # Set the model save path
    PATH = "model.pth"

    bs = args.batch_size  # Batch size
    num_epochs = args.epochs  # Number of epochs
    num_classes = 25  # Number of classes #11
    # Number of workers

    # Load data from folders
    transforms_dict = get_transform(args.transform_type)
    logger.info(f"transform type used is: {transforms_dict}")

    dataloaders, class_names = get_dataloader(
        train_directory=train_directory,
        transforms_dict=transforms_dict,
        args=args,
        logger=logger,
    )

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = get_model(args, num_classes, logger=logger)

    # Transfer the model to GPU
    model_ft = model_ft.to(device)

    logger.info("Model Summary:-\n")
    for num, (name, param) in enumerate(model_ft.named_parameters()):
        logger.info(
            f"Index: {num:3d} Layer name: {name:40s} requires grad: {param.requires_grad}"
        )
    summary(model_ft, input_size=(3, 224, 224))
    # logger.info(model_ft)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    logger.info(f"Optimizer settings: {optimizer_ft}")

    # Learning rate decay
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    logger.info(f"exp_lr_scheduler Step Size: {exp_lr_scheduler.step_size}")
    logger.info(f"exp_lr_scheduler Gamma: {exp_lr_scheduler.gamma}")
    # Model training routine

    def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = -10.0

        history = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        # Tensorboard summary
        writer = SummaryWriter(log_dir=output_dir)
        best_acc_epoch = -1

        for epoch in range(num_epochs):

            logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
            logger.info("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "valid", "test"]:
                since = time.time()

                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                num_images = 0.0
                all_preds = np.array([])
                all_labels = np.array([])
                # Iterate over data.
                for index, (inputs, labels) in enumerate(dataloaders[phase]):

                    if isinstance(inputs, dict):
                        inputs = inputs["image"]

                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if epoch == 0 and index == 0:
                        save_image_batch(inputs, phase, output_dir)
                        # exit()

                    num_images += inputs.shape[0]
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase != "train":
                        all_preds = np.concatenate(
                            (all_preds, preds.cpu().numpy()), axis=None
                        )
                        all_labels = np.concatenate(
                            (all_labels, labels.data.cpu().numpy()), axis=None
                        )

                    if index % 10 == 0:
                        logger.info(
                            "Split: {}, Epoch: {}/{}, Batch: {}/{} Loss: {:.4f}, Acc: {:.4f}%".format(
                                phase,
                                epoch + 1,
                                num_epochs,
                                index + 1,
                                len(dataloaders[phase]),
                                running_loss / num_images,
                                100 * running_corrects / num_images,
                            )
                        )

                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / num_images
                epoch_acc = running_corrects / num_images

                if isinstance(epoch_acc, torch.Tensor):
                    epoch_acc = epoch_acc.item()

                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

                logger.info(
                    "{}, Epoch:{}, Images: {}, Loss: {:.4f}, Acc: {:.4f}%".format(
                        phase, epoch, num_images, epoch_loss, epoch_acc * 100
                    )
                )

                # Record training loss and accuracy for each phase
                if phase == "train":
                    writer.add_scalar("Train/Loss", epoch_loss, epoch)
                    writer.add_scalar("Train/Accuracy %: ", epoch_acc * 100, epoch)
                    writer.flush()
                else:
                    writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
                    writer.add_scalar(f"{phase}/Accuracy %: ", epoch_acc * 100, epoch)
                    writer.flush()

                # deep copy the model
                if phase == "valid" and epoch_acc * 100 > best_acc:
                    best_acc = epoch_acc * 100
                    best_acc_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # checkpoint_path = os.path.join(
                    #     output_dir, f"model_epoch_{epoch}.pth"
                    # )

                    checkpoint_path = os.path.join(
                        output_dir, f"best_model_checkpoint.pth"
                    )

                    logger.info(
                        f"Saving the current best model of acc: {epoch_acc*100}% to"
                        f"path: {checkpoint_path}"
                    )

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            #'optimizer_state_dict': optimizer.state_dict(),
                            #'scheduler_state_dict': scheduler.state_dict(),
                            #'loss': epoch_loss,
                            "accuracy": epoch_acc * 100,
                        },
                        checkpoint_path,  # "../checkpoints/model_epoch_{}.pth".format(epoch),
                    )

                if phase != "train":
                    # cm = confusion_matrix(all_labels, all_preds)
                    draw_confusion_matrix(
                        ground_truth=all_labels,
                        predictions=all_preds,
                        num_classes=num_classes,
                        output_dir=output_dir,
                        epoch=epoch,
                        class_names=class_names,
                    )
                    # logger.info(f"Confusion matrix for {phase} phase:")
                    # logger.info(cm)

                time_elapsed = time.time() - since

                logger.info(
                    "Time {:.0f} mins {:.0f} secs".format(
                        time_elapsed // 60, time_elapsed % 60
                    )
                )

            # print(history)
            with open(os.path.join(output_dir, "training_history.json"), "w") as f:
                json.dump(history, f)

            logger.info(
                "Best val Acc so far: {:4f}% achieved "
                "at the end of round: {} and Test Accuracy is {:4f}%".format(
                    best_acc, best_acc_epoch, history["test_acc"][best_acc_epoch] * 100
                )
            )

            print()

        time_elapsed = time.time() - since

        logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        logger.info("Best val Acc: {:4f}%".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Train the model
    model_ft = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs
    )
    # Save the entire model
    print("\nSaving the model...")
    torch.save(model_ft, PATH)


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

    # args = vars(ap.parse_args())
    args = ap.parse_args()

    main(args)
