# MS_project

## Overview

This repository contains the baseline code for the Greek letter classification using `ResNet18/50` as the image classifier. The project is based on the `ICDAR2023` Competition on Detection and Recognition of `Greek Letters` on Papyri dataset.

## Dataset

We utilize the `ICDAR2023` Competition dataset for our experiments. The training set consists of `152` full images with annotations, while the test set contains `38` images without annotations. Additionally, we have generated `34,061` cropped images by cropping the letter images using ground-truth annotations. And We used Alpub dataset for pretraining, which contains 205,797 cropped images.

For our experiments, we split the cropped training images into a ratio of `70%`-`15%`-`15%` for `training`, `validation`, and `testing` purposes, respectively.


## Code Structure

```
Baseline/
│
├── src/
│   ├── hpo_v1.py
|   ├── train.py
│   ├── train.py
│   └── nets.py
│
├── scripts/
|   / search_space
│   └── hpo_resnet18_ce.sh
│
└── README.md
```

```
simclr/
│
├── src/
│   ├── hpo_v2.py
|   ├── simclr.py
│   ├── run.py
│   └── parse_args.py
│
├── scripts/
|   / search_space
│   └── hpo_resnet18_ce.sh
│
└── README.md
```


## Usage

To train the `Baseline model`, please run the following command:

```
bash Baseline/scripts/hpo_resnet18_4_com_ce_alpub_seed2_sorted_4aug.sh
```

To evaluate the `Baseline model`, please run the following command:

```
bash Baseline/scripts/hpo_resnet18_4_com_ce_alpub_seed2_sorted_finetune_4aug.sh
```

Example of the baseline model bash code(pertaining): 
```
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=20
hpo_exp_name="CE_loss_resnet18_4_aug_alpub_seed2_sorted_4augs_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python Baseline/src/hpo_v1.py \
        --mode finetune \
        --arch resnet18 \
        --num_classes 24 \
        --batch_size 198  \
        --epochs ${epochs} \
        --dataset alpub \
        --gpu_per_worker 0.5 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn ce \
        --triplet_embedding_size 64 \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights True \
        --hpo_search_space_config "scripts/search_space/search_space_4_com_alpub_ce_seed2_sorted_4augs.yaml" $@ >& ./logs/${log_filepath} 
```
For fine-tuning we have changed only `--num_classes` to 24, `--dataset` to alpub and `--use_imagenet_pretrained_weights` to `False`. Make sure to check filename, `hpo_search_space_config` file

--------------------------------------------------------------------------------------------------------------------------------------------
To train the `Triplet model`, please run the following command:

```
bash Baseline/scripts/hpo_resnet18_triplet_4combinations_alpub_seed2.sh
```

To evaluate the `Triplet model`, please run the following command:

```
bash Baseline/scripts/hpo_resnet18_triplet_finetune_4combinations_alpub.sh
```

Example of Triple model bash code
```
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=20
hpo_exp_name="Triplet_loss_alpub_4_agumentations_resnet18_seed2_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python Baseline/src/hpo_v1.py \
        --mode finetune \
        --arch resnet18 \
        --num_classes 24 \
        --batch_size 48  \
        --epochs ${epochs} \
        --dataset alpub \
        --gpu_per_worker 1.0 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn triplet \
        --triplet_embedding_size 64 \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights True \
        --hpo_search_space_config "scripts/search_space/search_space_4_combinations_alpub_triplet_seed2.yaml" $@ >& ./logs/${log_filepath} 
```
For fine-tuning we have changed only `--num_classes` to 24, `--dataset` to alpub and `--use_imagenet_pretrained_weights` to `False`. Make sure to check file name, `hpo_search_space_config` file

--------------------------------------------------------------------------------------------------------------------------------------------
To train the `SimCLR model`, please run the following command:

```
bash simclr/scripts/hpo_resnet18_resize198_top4_combinations_with_seed2.sh
```

To evaluate the `SimCLR model`, please run the following command:

```
bash simclr/scripts/hpo_resnet18_simclr_finetune_alpub_4combinations_seed2.sh
```

SExample of SimCLR model bash code (pertaining):
```
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=100


dataset=alpub

#hpo_exp_name="resize64_65_combinations"
hpo_exp_name="${dataset}_simclr_resize198_top4_combinations_with_198_seed2"
log_filepath=${hpo_exp_name}_$(date +'%Y-%m-%d_%H-%M-%S').log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python simclr/src/hpo_v2.py \
        --dataset_name ${dataset} \
        --num_classes 24 \
        --arch resnet18 \
        --batch_size 164  \
        --epochs ${epochs} \
        --log-every-n-steps 100 \
        --gpu-index 2 \
        --lr 0.0003 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --gpu_per_worker 1.0 \
        --cpu_per_worker 2 \
        --out_dim  128 \
        --launch_script_path ${SCRIPT_PATH} \
        --hpo_search_space_config "scripts/search_space/search_space_resize198_top4_combinations_with_seed2.yaml" $@ >& ./logs/${log_filepath}
```
For fine-tuning we have generated ckpt_path_dict file during training. We have added this file in Baseline code for fine-tuning.

Below is an example of SimCLR model fine-tuning bash code:
```
epochs=20
hpo_exp_name="alpub_SimCLR_finetine_with_ce_loss_unfreeze_backbone_4_agumentations_198_seed2_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python Baseline/src/hpo_v1.py \
        --mode simclr_finetune_with_ce \
        --num_classes 25 \
        --arch resnet18 \
        --batch_size 128  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 0.5 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn simclr_finetune_with_ce \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights False \
        --freeze_backbone False \
        --ckpt_path_dict /home/vedasri/SimCLR_V2/results_hpo/final_experiments/alpub_simclr_resize198_top4_combinations_with_198_seed2/config2ckpt_path.json \
        --hpo_search_space_config "scripts/search_space/search_space_4_combinations_alpub_simclr_seed2.yaml" $@ >& ./logs/${log_filepath} 
```


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Below are the 3 commands to generate augmented views of 3 models:

Baseline model:
```
python ./Baseline/src/core/datatransform_builder_v3.py --transform_type randomcrop224,gaussianblur,gray
```
Triplet model:
```
python ./Baseline/src/core/datatransform_builder_v3_triplet.py --transform_type randomcrop224 --loss_fn triplet
```
SimCLR model:
```
python .simclr/src/data_aug/datatransform_visuals.py --transform_type randomcrop198,morpho_dilation
```



## Results

We have devided results into 2 parts. One is `Without pre-training on Alpub`, second `With pre-training on Alpub`

- After training for `10` epochs, we obtain a training accuracy of `79.42%`, Validation accuracy of `76.38%` and Test accuracy of `75.49%`.


