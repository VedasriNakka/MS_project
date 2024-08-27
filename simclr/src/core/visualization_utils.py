from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision
import torch


def draw_confusion_matrix(
    ground_truth, predictions, num_classes, output_dir, epoch, class_names
):

    if not os.path.exists(os.path.join(output_dir, "conf_mats")):
        os.makedirs(os.path.join(output_dir, "conf_mats"))

    confusion_matrix = metrics.confusion_matrix(
        ground_truth, predictions, labels=np.arange(num_classes)
    )
    plt.figure(figsize=(12, 12))

    # define and print matrix with labels
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="g",
        xticklabels=class_names,
        yticklabels=class_names,  # ["Not Spam", "Spam"],
    )

    # display matrix
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Prediction", fontsize=12)
    plt.savefig(os.path.join(output_dir, "conf_mats", f"epoch={epoch}_cm.jpg"))


def save_image_batch_old(batch_data, phase, output_dir):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))

    def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    # tensor_x: size [B, 3, H, W]
    torchvision.utils.save_image(
        denormalize(batch_data), os.path.join(output_dir, f"batch_data_{phase}.png")
    )
    # logger.info("saved batch data in the image format to the output folder!")
