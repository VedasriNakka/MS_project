from torchvision import datasets, models, transforms
import logging
import json
import os

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
import torch.utils.data as data
import multiprocessing
import torch
from torch.utils.data import Dataset
import numpy as np

import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import random

from data_aug.view_generator import ContrastiveLearningViewGenerator


def create_class_to_idx(root_dir):
    classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    with open(os.path.join(root_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)

    return class_to_idx


class BaseDataset(Dataset):
    def __init__(
        self, file_paths, transform=None, root_dir=None, num_classes=25, n_views=2
    ):
        self.file_paths = file_paths
        # self.transform = transform
        self.classname2index = create_class_to_idx(root_dir)
        self.index2classname = {v: k for k, v in self.classname2index.items()}
        self.classnames = [self.index2classname[i] for i in range(num_classes)]
        self.transform = ContrastiveLearningViewGenerator(transform, n_views)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")

        image = np.array(image)

        label = os.path.basename(os.path.dirname(img_path))
        label = self.classname2index[label]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image=image)
            # image = self.transform(image=image)

        return image, label


def get_dataloader_v2(
    train_directory,
    transforms_dict,
    args,
    logger,
    shuffle_train=True,
    add_manual_seed=False,
    num_classes=None,
):

    num_cpu = multiprocessing.cpu_count()

    # Load splits from JSON file
    with open(os.path.join(train_directory, "data_splits.json"), "r") as f:
        splits = json.load(f)

    if args.loss_fn == "info_nce":

        train_dataset = BaseDataset(
            file_paths=splits["train"],
            transform=transforms_dict["train"],
            root_dir=train_directory,
            num_classes=num_classes,
        )
        val_dataset = BaseDataset(
            file_paths=splits["val"],
            transform=transforms_dict["valid"],
            root_dir=train_directory,
            num_classes=num_classes,
        )
        test_dataset = BaseDataset(
            file_paths=splits["test"],
            transform=transforms_dict["test"],
            root_dir=train_directory,
            num_classes=num_classes,
        )
    else:
        raise NotImplementedError()

    class_names = train_dataset.classnames

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Create iterators for data loading
    dataloaders = {
        "train": data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            num_workers=num_cpu,
            pin_memory=True,
            drop_last=False,
        ),  # dataset['train']
        "valid": data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_cpu,
            pin_memory=True,
            drop_last=False,
        ),
        "test": data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_cpu,
            pin_memory=True,
            drop_last=False,
        ),
    }

    return dataloaders, class_names
