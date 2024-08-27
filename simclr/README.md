##  SimCLR: MT_augmentation_and_contrastive_learning

## Overview

This repository contains the code for the Greek letter classification using Contrastive Learning with `ResNet18/34/50` as the image classifier. The project is based on the `ICDAR2023` Competition on Detection and Recognition of `Greek Letters` on Papyri dataset.

## Dataset

We utilize the `ICDAR2023` Competition dataset for our experiments. The training set consists of `152` full images with annotations, while the test set contains `38` images without annotations. Additionally, we have generated `35,597` cropped images by cropping the letter images using ground-truth annotations. 

For our experiments, we split the cropped training images into a ratio of `70%`-`15%`-`15%` for `training`, `validation`, and `testing` purposes, respectively.


## Code Structure

```
Baseline/
│
├── src/
│   ├── run.py
│   ├── simclr.py
│   └── utils.py
│
├── scripts/
│   ├── train_resnet34.sh
│   ├── train_resnet50.sh
│   └── train_resnet18.sh
│
└── README.md
```


## Usage

To train and also to evaluate the model, please run the following command:

```
bash scripts/train_resnet18.sh
bash scripts/train_resnet34.sh
bash scripts/train_resnet50.sh
```

In particular, we provide multiple commandl-line arguments such as batch size, number of epochs, model architecture. Our script looks as below:

```
python -W ignore src/run.py -data ../datasets -dataset-name HomerCompTraining_Cropped \
        --log-every-n-steps 100 --epochs 100 --gpu-index 2 --lr 0.0003 --arch resnet18 --batch-size 8 --seed 2
```

## Results

- After training for `XX` epochs, we obtain a training accuracy of `XX%`, Validation accuracy of `XX%` and Test accuracy of `XX%`.


