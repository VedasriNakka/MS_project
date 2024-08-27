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

from src.core.datasets.base_dataset import BaseDataset, TripletDataset

# logger.propagate = True


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):

        image = np.array(self.dataset[index][0])

        # print(self.dataset[index][0])
        if self.transform:
            x = self.transform(image=image)
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)
    


def get_dataset(args, train_directory):



    if args.loss_fn == "ce":
         # Load data from folders with full transformation for train
        train_dataset_full = datasets.ImageFolder(
            root=train_directory,
            # transform=transforms_dict["train"]
        
        )

        return train_dataset_full
    
    elif args.loss_fn  == "triplet":

        train_dataset_full = None
        return train_dataset_full

    else:
        raise NotImplementedError()











def get_dataloader(
    train_directory,
    transforms_dict,
    args,
    logger,
    shuffle_train=True,
    add_manual_seed=False,
):

    num_cpu = multiprocessing.cpu_count()

    
    
    
    # train_dataset_full = datasets.ImageFolder(
    #             root=train_directory,
    #             # transform=transforms_dict["train"]
    #         )

    train_dataset_full = get_dataset(args= args, train_directory=train_directory)
    

   

    # Class names or target labels
    class_names = train_dataset_full.classes        #commenting this 
    # logger.info(f"Classes:, {class_names}")

    logger.info(f"Training full dataset size: {len(train_dataset_full)}")
    num_classes = len(train_dataset_full.classes)

    # print(num_classes)
    # exit()

    # Take a subset of the full dataset (e.g., first 5000 samples)
    # train_dataset_subset = torch.utils.data.Subset(train_dataset_full, range(3000))

    from torch.utils.data import random_split

    # Define the sizes of each split (e.g., 70% train, 15% validation, 15% test)
    train_size = int(0.7 * len(train_dataset_full))
    val_size = int(0.15 * len(train_dataset_full))
    test_size = len(train_dataset_full) - train_size - val_size

    if add_manual_seed:
        torch.manual_seed(42)

    # Use random_split to split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset_full,
        [train_size, val_size, test_size],
    )

    train_dataset = MyLazyDataset(train_dataset, transforms_dict["train"])
    val_dataset = MyLazyDataset(val_dataset, transforms_dict["valid"])
    test_dataset = MyLazyDataset(test_dataset, transforms_dict["test"])

    # train_dataset.dataset.transform = transforms_dict["train"]
    # val_dataset.dataset.transform = transforms_dict["valid"]
    # test_dataset.dataset.transform = transforms_dict["test"]
    # print("Train transform: ", train_dataset.transform)
    # exit()

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Size of train and validation data
    dataset_sizes = {
        "train": len(train_dataset),
        "valid": len(val_dataset),
        "test": len(test_dataset),
    }

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








def get_dataloader_v2(
    train_directory,
    transforms_dict,
    args,
    logger,
    shuffle_train=True,
    add_manual_seed=False,
    num_classes = None
    ):

    num_cpu = multiprocessing.cpu_count()


    # Load splits from JSON file
    with open(os.path.join(train_directory, 'data_splits.json'), 'r') as f:
        splits = json.load(f)

    if args.loss_fn == "ce" or args.loss_fn == "triplet_finetune_with_ce" or args.loss_fn == "simclr_finetune_with_ce"  :

        train_dataset = BaseDataset(file_paths=splits['train'], transform=transforms_dict["train"], root_dir=train_directory, num_classes=num_classes)
        val_dataset = BaseDataset(file_paths=splits['val'], transform=transforms_dict["valid"], root_dir=train_directory, num_classes=num_classes)
        test_dataset = BaseDataset(file_paths=splits['test'], transform=transforms_dict["test"], root_dir=train_directory, num_classes=num_classes)

    elif args.loss_fn == "triplet":

        
        train_dataset = TripletDataset(file_paths=splits['train'], transform=transforms_dict["train"], root_dir=train_directory, num_classes=num_classes)
        val_dataset = TripletDataset(file_paths=splits['val'], transform=transforms_dict["valid"], root_dir=train_directory, num_classes=num_classes)
        test_dataset = TripletDataset(file_paths=splits['test'], transform=transforms_dict["test"], root_dir=train_directory, num_classes=num_classes)

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
