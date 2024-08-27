import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os
import sys
import time
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import torchvision
from ray import train, tune

from sklearn.metrics import confusion_matrix
#from src.core.visualization_utils import draw_confusion_matrix, save_image_batch


torch.manual_seed(0)


def plot_tnse(features_all, labels_all, logger, log_dir, epoch, split):

    features = np.concatenate(features_all, 0)
    labels = np.concatenate(labels_all, 0)

    logger.info(f"features: {features.shape}, labels: {labels.shape}")
    
    mask = labels < 10
    features = features[mask]
    labels = labels[mask]

    logger.info(f"Filtered features: {features.shape}, labels: {labels.shape}")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(features)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10')

    # Add legend
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.title("t-SNE visualization of embeddings")
    # plt.xlabel("t-SNE feature 1")
    # plt.ylabel("t-SNE feature 2")
    plt.savefig(os.path.join(log_dir, f"{split}_tsne_{epoch}.png"))

    return 




class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.writer = SummaryWriter(log_dir=kwargs["output_dir"])
        # logging.basicConfig(
        #     filename=os.path.join(self.writer.log_dir, "training.log"),
        #     level=logging.DEBUG,
        # )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        logger.info("SimCLR training constructor is finished!")

    def info_nce_loss(self, features):

        # batch_size = self.args.batch_size
        batch_size = len(features) // 2

        labels = torch.cat(
            [torch.arange(batch_size) for i in range(self.args.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        
        return logits, labels

    def save_image_batch(self, batch_data, phase, index):
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
            os.path.join(self.writer.log_dir, f"batch_data_{phase}_{index}.png"),
        )
        # logger.info("saved batch data in the image format to the output folder!")

    def train(self, train_loader, valid_loader, test_loader):
        since = time.time()

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)
        logger.info("\n\n\n")

        n_iter = 0
        logger.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logger.info(f"Training with gpu: {self.args.device}.")

        self.model = self.model.to(self.args.device)

        best_val_acc = 0.0
        best_test_acc = 0.0

        for epoch_counter in range(self.args.epochs):

            logger.info(
                f"---------- Epoch {epoch_counter+1}/{self.args.epochs} ----------------------------"
            )

            running_loss = 0.0
            num_images = 0
            num_total_correct = 0

            self.model.train()

            epoch_metrics = {}

            features_all = []
            labels_all = []
            isplot_tnse =  True



            for index, (images, gt_labels) in enumerate(train_loader):

                # since = time.time()
                #logger.info(f"Class labels: {class_label}")
                # if isinstance(images[0], dict):

                images = [img["image"] for img in images]
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                if epoch_counter == 0 and index < 20:
                    self.save_image_batch(images, "train", index)
                # exit()

                # logger.info(f"images shape: {images.shape[0]}")

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)

                    # logger.info(f"labels shape: {labels.shape}") # [256]
                    # logger.info(f"labels size: {len(labels)}")
                    # logger.info(f"labels: {labels}")

                    # logger.info(f"logits shape: {logits.shape}")    # [256, 255]
                    # logger.info(f"logits size: {len(logits)}")
                    # logger.info(f"logits: {logits}")

        
                    loss = self.criterion(logits, labels)

                if False:
                    if len(features_all) < 1000//self.args.batch_size:
                            features_all.append(features[:len(gt_labels)].detach().cpu().numpy())
                            labels_all.append(gt_labels.detach().cpu().numpy())

                    elif isplot_tnse:
                        plot_tnse(features_all, labels_all, logger, self.writer.log_dir, epoch_counter,split="train")
                        isplot_tnse = False


                self.optimizer.zero_grad()

                # we dont update the weights for the first epoch to plot the tsne
                if epoch_counter > 0:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                running_loss += loss.item() * images.size(0)
                num_images += images.shape[0]

                #logger.info(f"logits; {logits.shape}, labels: {labels.shape}")

                top1, _ = accuracy(logits, labels, topk=(1, 1))
                num_total_correct += top1[0] * images.shape[0] / 100.0

                if index % 10 == 0:
                    logger.info(
                        "Split: {}, Epoch: {}/{}, Batch: {}/{} Loss: {:.4f}, Acc: {:.4f}%".format(
                            "Train",
                            epoch_counter + 1,
                            self.args.epochs,
                            index + 1,
                            len(train_loader),
                            running_loss / num_images,
                            100 * num_total_correct / num_images,  # top1[0],
                        )
                    )

                n_iter += 1

                # time_elapsed = time.time() - since

                # logger.info(
                #     "Training complete in {:.0f}m {:.0f}s".format(
                #         time_elapsed // 60, time_elapsed % 60
                #     )
                # )

            self.writer.add_scalar("loss", running_loss / num_images, epoch_counter)
            self.writer.add_scalar(
                "acc/top1", 100 * num_total_correct.item() / num_images, epoch_counter
            )
            self.writer.add_scalar(
                "learning_rate", self.scheduler.get_lr()[0], global_step=n_iter
            )
            self.writer.flush()

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
            )
            logger.info(
                f"Epoch:{epoch_counter+1}/{self.args.epochs},\t Agg. Loss: {loss:.4f}, \t Agg. Top1 Acc: {100*num_total_correct/num_images:.5f}%\n\n"
            )

            # ------------------------------------------------------

            curr_metrics = {}
            epoch_metrics["train_acc"] = (100 * num_total_correct / num_images).item()
            epoch_metrics["train_loss"] = running_loss / num_images

            for split, data_loader in zip(
                ["valid", "test"], [valid_loader, test_loader]
            ):

                since = time.time()

                running_loss = 0.0
                num_images = 0
                num_total_correct = 0
                curr_acc = 0.0

                self.model.eval()

                features_all = []
                labels_all = []
                isplot_tnse =  True

                for index, (images, gt_labels) in enumerate(data_loader):

                    # draw_confusion_matrix(
                    #     ground_truth=num_images,
                    #     predictions=num_total_correct,
                    #     num_classes=num_classes,
                    #     output_dir=output_dir,
                    #     epoch=epoch,
                    #     class_names=class_names,
                    # )

                    images = [img["image"] for img in images]

                    images = torch.cat(images, dim=0)
                    images = images.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        features = self.model(images)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)


                    if split in ["test"]:                       
                        if len(features_all) < 2000//self.args.batch_size:
                            features_all.append(features[:len(gt_labels)].detach().cpu().numpy())
                            labels_all.append(gt_labels.detach().cpu().numpy())

                        elif isplot_tnse:
                            plot_tnse(features_all, labels_all, logger, self.writer.log_dir, epoch_counter, split)
                            isplot_tnse = False





                    running_loss += loss.item() * images.size(0)
                    num_images += images.shape[0]
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    # accuracy in percentages
                    num_total_correct += top1[0] * images.shape[0] / 100.0

                    curr_acc = 100 * num_total_correct / num_images

                    epoch_loss = running_loss / num_images
                    epoch_acc = 100 * num_total_correct / num_images

                    if index % 10 == 0:
                        logger.info(
                            "Split: {}, Epoch: {}/{}, Batch: {}/{},  Loss: {:.4f}, Acc: {:.4f}%".format(
                                split,
                                epoch_counter + 1,
                                self.args.epochs,
                                index + 1,
                                len(data_loader),
                                running_loss / num_images,
                                curr_acc,
                            )
                        )

                        # print(num_total_correct, num_images)

                    n_iter += 1

                time_elapsed = time.time() - since

                logger.info(
                    "Training complete in {:.0f}m {:.0f}s".format(
                        time_elapsed // 60, time_elapsed % 60
                    )
                )

                self.writer.add_scalar(
                    f"{split} loss", running_loss / num_images, epoch_counter
                )
                self.writer.add_scalar(
                    f"{split} acc/top1", curr_acc.item(), epoch_counter
                )
                self.writer.flush()

                epoch_metrics[f"{split}_loss"] = running_loss / num_images
                epoch_metrics[f"{split}_acc"] = curr_acc.item()

                curr_metrics[f"{split}_acc"] = curr_acc
                curr_metrics[f"{split}_loss"] = running_loss / num_images

                logger.info(
                    f"Split: {split}, Epoch:{epoch_counter+1}/{self.args.epochs},\t Agg. Loss: {running_loss/num_images:.4f}, "
                    f"\t Agg. Top1 Acc: {curr_acc:.4f}%,\t\n"
                )

            logger.info(f"metric is {epoch_metrics}")
            train.report(
                metrics={**epoch_metrics},
                # checkpoint=checkpoint,
            )

            if curr_metrics["valid_acc"] > best_val_acc:

                best_val_acc = curr_metrics["valid_acc"]
                best_test_acc = curr_metrics["test_acc"]
                best_val_acc_epoch = epoch_counter + 1

                checkpoint_name = "best_checkpoint.pth.tar".format(self.args.epochs)
                save_checkpoint(
                    {
                        "epoch": epoch_counter + 1,
                        "arch": self.args.arch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "accu_valid": best_val_acc,
                    },
                    is_best=True,
                    filename=os.path.join(self.writer.log_dir, checkpoint_name),
                )
                logging.info(
                    f"Best Model checkpoint achieved at epoch {epoch_counter+1} has been saved at {os.path.join(self.writer.log_dir, checkpoint_name)}."
                )

            logging.info(
                "By the end of {}/{} epochs, Best validation accuracy: {:.5f}% "
                "achieived at epoch: {} and corresponding test accuracy is : {:.5f}%".format(
                    epoch_counter + 1,
                    self.args.epochs,
                    best_val_acc,
                    best_val_acc_epoch,
                    best_test_acc,
                )
            )

            logger.info(
                "----------------------------------------------------------------------------------\n\n"
            )
            time_elapsed = time.time() - since

            logger.info(
                "Training complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = "final_checkpoint.pth.tar"
        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name),
        )
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.writer.log_dir}."
        )
