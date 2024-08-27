import time
import copy
import os
import sys
import numpy as np
import torch
import json
from ray import train, tune
#from src.core.visualization_utils import draw_confusion_matrix, save_image_batch
from src.core.visualization_utils_v2 import draw_confusion_matrix, save_image_batch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', edgecolor="black")

    # Add legend
    plt.legend(*scatter.legend_elements(), title="Classes", fancybox=True, shadow=True)

    plt.title("t-SNE visualization of embeddings")
    # plt.xlabel("t-SNE feature 1")
    # plt.ylabel("t-SNE feature 2")
    plt.savefig(os.path.join(log_dir, f"{split}_tsne_{epoch}.png"), dpi=100, bbox_inches="tight")

    return 



# Triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
    distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean(), distance_positive, distance_negative


# Triplet accuracy function
def triplet_accuracy(distance_positive, distance_negative):
    return (distance_positive < distance_negative).float().mean()


# Triplet accuracy function
def count_good_triplet(distance_positive, distance_negative):
    return (distance_positive < distance_negative).float().sum()


def train_model_triplet(model, criterion, optimizer, scheduler, num_epochs=30, 
                num_classes=None, class_names=None, 
                dataloaders=None, device=None, 
                SummaryWriter=None, output_dir=None, logger=None, args=None):
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

        epoch_metrics = {}

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

            all_features = []
            all_gt_labels = []
            isplot_tnse = True


            for index, (anchor, positive, negative, labels) in enumerate(dataloaders[phase]):

                # if isinstance(inputs, dict):
                #     inputs = inputs["image"]

                anchor = anchor.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)

                labels = labels.to(device, non_blocking=True)

                if epoch == 0 and index == 0:
                    save_image_batch(anchor, "anchor_"+ phase, output_dir)
                    save_image_batch(positive, "positive_"+ phase, output_dir)
                    save_image_batch(negative, "negative_"+ phase, output_dir)

                    # exit()

                num_images += anchor.shape[0]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    anchor_output_dict = model(anchor)

                    #logger.info(f"anchor output {anchor_output.shape}")

                    positive_output_dict = model(positive)
                    negative_output_dict = model(negative)
                    loss, distance_positive, distance_negative = triplet_loss(anchor_output_dict["features"], positive_output_dict["features"], negative_output_dict["features"])
                    num_correct_triplets = count_good_triplet(distance_positive, distance_negative)
                    
                    
                    #logger.info(f"Distances; {distance_positive}, images lenth; {anchor.shape[0]}")

                    # backward + optimize only if in training phase and for the first epoch, model is not 
                    # updated
                    if phase == "train" and epoch > 0:
                        loss.backward()
                        optimizer.step()

                # plot tsne 
                if phase in ["test",]:
                    features = anchor_output_dict["features"]

                    if len(all_features) < 2000//args.batch_size:
                        all_features.append(features[:len(labels)].detach().cpu().numpy())
                        all_gt_labels.append(labels.detach().cpu().numpy())

                    elif isplot_tnse:
                        plot_tnse(all_features, all_gt_labels, logger, output_dir, epoch, phase)
                        isplot_tnse = False
                        break


                # statistics
                running_loss += loss.item() * anchor.size(0)
                running_corrects += num_correct_triplets

                # if phase != "train":
                #     all_preds = np.concatenate(
                #         (all_preds, preds.cpu().numpy()), axis=None
                #     )
                #     all_labels = np.concatenate(
                #         (all_labels, labels.data.cpu().numpy()), axis=None
                #     )

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

            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc * 100

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
            if phase == "valid":
                if epoch_acc * 100 > best_acc:
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
                    epoch_metrics[f"found_best_model"] = True
                else:
                    epoch_metrics[f"found_best_model"] = False


            if phase != "train":
                # cm = confusion_matrix(all_labels, all_preds)
                draw_confusion_matrix(
                    ground_truth=all_labels,
                    predictions=all_preds,
                    num_classes=num_classes,
                    output_dir=output_dir,
                    epoch=epoch,
                    class_names=class_names,
                    data_loader= dataloaders[phase],
                    phase = phase

                )
                # logger.info(f"Confusion matrix for {phase} phase:")
                # logger.info(cm)

            time_elapsed = time.time() - since

            logger.info(
                "Time {:.0f} mins {:.0f} secs".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

        train.report(
            metrics={**epoch_metrics},
            # checkpoint=checkpoint,
        )

        # print(history)
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f)

        logger.info(
            "Best val Acc so far: {:4f}% achieved "
            "at the end of round: {} and Triplet Test Accuracy is {:4f}%".format(
                best_acc, best_acc_epoch, history["test_acc"][best_acc_epoch] * 100
            )
        )

        print()

    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "accuracy": epoch_acc * 100,
        },
        os.path.join(
        output_dir, f"model_checkpoint_epoch={epoch}.pth")
    )

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