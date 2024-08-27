from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision
import torch
import json 
import torch.utils.data as data
from src.core.dataloader_builder import get_dataloader
from src.core.data_builder import get_dataset

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def draw_confusion_matrix(
    ground_truth, predictions, num_classes, output_dir, epoch, class_names, data_loader=None, phase="test", 
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
    plt.savefig(os.path.join(output_dir, "conf_mats", f"epoch={epoch}_split={phase}_cm.jpg"))
    plt.close()

    # Analyze the confusion matrix to find top confusions
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                confusion_pairs.append((confusion_matrix[i, j], class_names[i], class_names[j]))

    # Sort by number of confusions (highest first)
    confusion_pairs.sort(reverse=True, key=lambda x: x[0])
    # Extract top 3 and top 5 confusions
    top10_confusions = confusion_pairs[:10]
    top10_confusions_dict = {}

    logger.info("\nTop 10 Confusions:")
    for index, (count, actual, predicted) in enumerate(top10_confusions):
        logger.info(f"Actual: {actual}, Predicted: {predicted}, Count: {count}")
        top10_confusions_dict[f"top_{index}"] = {"actual": actual, "predicted": predicted, "count": int(count)}

    
    with open(os.path.join(output_dir, "conf_mats", f"epoch={epoch}_split={phase}_top10.json"), "w") as f:
        json.dump(top10_confusions_dict, f, indent=4)

    
    if False:
        # # Create a DataLoader for the dataset
        # data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        # Collect top 3 confusing data points

        top_confusing_data_points = []
        for idx, (data, label) in enumerate(data_loader):
            pred = predictions[idx]
            if len(top_confusing_data_points) < 3:
                for count, actual, predicted in top3_confusions:
                    if class_names[label.item()] == actual and class_names[pred] == predicted:
                        top_confusing_data_points.append((data, actual, predicted, label.item(), pred))
                        if len(top_confusing_data_points) == 3:
                            break

        # Create an image to display the top 3 confusing data points
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (data, actual, predicted, label, pred) in enumerate(top_confusing_data_points):
            ax = axes[i]
            img = data.squeeze().numpy().transpose(1, 2, 0)
            IMG_MEAN = [0.485, 0.456, 0.406]
            IMG_STD = [0.229, 0.224, 0.225]
            # Denormalize the image
            img_denorm = (img * IMG_STD) + IMG_MEAN

            # Clip the values to be in the valid range [0, 1]
            img_denorm = np.clip(img_denorm, 0, 1)

            #img = np.clip((img * 0.5) + 0.5, 0, 1)  # Denormalize
            ax.imshow(img_denorm)
            ax.set_title(f"Actual: {actual}\nPredicted: {predicted}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "conf_mats", f"epoch={epoch}_split={phase}_top_confusing_data_points.jpg"))
        plt.close()


def save_image_batch(batch_data, phase, output_dir):
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
