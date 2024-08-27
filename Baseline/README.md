##  Baseline Method: MT_augmentation_and_contrastive_learning

## Overview

This repository contains the baseline code for the Greek letter classification using `ResNet18/34/50` as the image classifier. The project is based on the `ICDAR2023` Competition on Detection and Recognition of `Greek Letters` on Papyri dataset.

## Dataset

We utilize the `ICDAR2023` Competition dataset for our experiments. The training set consists of `152` full images with annotations, while the test set contains `38` images without annotations. Additionally, we have generated `35,597` cropped images by cropping the letter images using ground-truth annotations. 

For our experiments, we split the cropped training images into a ratio of `70%`-`15%`-`15%` for `training`, `validation`, and `testing` purposes, respectively.


## Code Structure

```
Baseline/
│
├── src/
│   ├── train.py
│   ├── train.py
│   └── nets.py
│
├── scripts/
│   └── train.sh
│
└── README.md
```


## Usage

To train and also to evaluate the model, please run the following command:

```
bash scripts/train.sh
```

In particular, we provide multiple commandl-line arguments such as batch size, number of epochs, model architecture. Our script looks as below:

```
epochs=50
python src/train.py --mode finetune --arch resnet18 --batch_size 64 --epochs ${epochs}
python src/train.py --mode finetune --arch resnet34 --batch_size 64 --epochs ${epochs}
python src/train.py --mode finetune --arch resnet50 --batch_size 64 --epochs ${epochs}
```

## Results

- After training for `10` epochs, we obtain a training accuracy of `79.42%`, Validation accuracy of `76.38%` and Test accuracy of `75.49%`.


