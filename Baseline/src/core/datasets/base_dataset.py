import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import random

def create_class_to_idx(root_dir):
    classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    with open(os.path.join(root_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=4)

    return class_to_idx

class BaseDataset(Dataset):
    def __init__(self, file_paths, transform=None, root_dir=None, num_classes=25):
        self.file_paths = file_paths
        self.transform = transform
        self.classname2index  = create_class_to_idx(root_dir)
        self.index2classname = {v:k for k,v in self.classname2index.items()}
        self.classnames  = [ self.index2classname[i] for i in range(num_classes)]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        image = np.array(image)

        label = os.path.basename(os.path.dirname(img_path))
        label = self.classname2index[label]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image=image)

        return image, label





class TripletDataset(Dataset):
    def __init__(self, file_paths, transform=None, root_dir=None, num_classes=25):
        self.file_paths = file_paths
        self.transform = transform
        self.classname2index = create_class_to_idx(root_dir)
        self.index2classname = {v: k for k, v in self.classname2index.items()}
        self.classnames = [self.index2classname[i] for i in range(num_classes)]
        
        # Create a dictionary to quickly access images of each class
        self.class_to_indices = {cls: [] for cls in self.classnames}
        for idx, img_path in enumerate(self.file_paths):
            label = os.path.basename(os.path.dirname(img_path))
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Get anchor image and its label
        anchor_img_path = self.file_paths[idx]
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        anchor_label_str = os.path.basename(os.path.dirname(anchor_img_path))
        anchor_label = self.classname2index[anchor_label_str]
        anchor_label = torch.tensor(anchor_label)

       
        # Select a positive image
        positive_indices = self.class_to_indices[anchor_label_str].copy()
        if len(positive_indices) > 1:
            positive_indices.remove(idx)
            positive_index = random.choice(positive_indices)
        else:
            positive_index = idx  # In this case, there's only one image, use the same image

        positive_img_path = self.file_paths[positive_index]
        positive_img = Image.open(positive_img_path).convert('RGB')
        
        # Select a negative image
        negative_label_str = anchor_label_str
        while True:
            negative_label_str = random.choice(self.classnames)
            if negative_label_str != anchor_label_str and self.class_to_indices[negative_label_str]:
                break

        #print("The negative label is ", negative_label_str)
        
        negative_index = random.choice(self.class_to_indices[negative_label_str])
        negative_img_path = self.file_paths[negative_index]
        negative_img = Image.open(negative_img_path).convert('RGB')
        negative_label = self.classname2index[negative_label_str]
        negative_label = torch.tensor(negative_label)

        if self.transform:
            anchor_img = self.transform(image=np.array(anchor_img))['image']
            positive_img = self.transform(image=np.array(positive_img))['image']
            negative_img = self.transform(image=np.array(negative_img))['image']

        return anchor_img, positive_img, negative_img, anchor_label