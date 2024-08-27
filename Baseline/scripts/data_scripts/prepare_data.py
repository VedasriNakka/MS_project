import os
import random
import json
from sklearn.model_selection import train_test_split

def generate_splits(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0

    # Collect all image paths and their corresponding labels
    image_paths = []
    labels = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_name)

    print(f"Number of images:  {len(image_paths)}")

    # Split the dataset into train, val, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, stratify=None, test_size=(val_ratio + test_ratio), random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, stratify=None, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
    )

    # Save the splits to JSON files
    splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }

    with open('./data_splits_alpub.json', 'w') as f:
        json.dump(splits, f, indent=4)

    return splits

#root_dir = '/home/vedasri/datasets/HomerCompTraining_Cropped'
root_dir = "/home/vedasri/datasets/alpub_v2/images"
splits = generate_splits(root_dir)
