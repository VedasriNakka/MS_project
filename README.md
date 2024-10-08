# Contrastive Learning for Character Detection in Ancient Greek Papyri
Completed project under Supervision of **Prof. Rolf Ingold**, **Prof. Andreas Fischer** and **Lars Vogtlin**

-- Vedasri Nakka

**Note**: If you like my findings and useful, please cite my paper: http://arxiv.org/abs/2409.10156

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

We have used 93 augmentations for without pretraining on Alpub dataset (on ResNet18, resNet50 architectures). Those are available in file
```
./Baseline_V2/scripts/search_space/search_space_subset_93_demo.yaml
````

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
For fine-tuning we have changed only `--num_classes` to 25, `--dataset` to icdar, and `--use_imagenet_pretrained_weights` to `False`. Make sure to check filename, `hpo_search_space_config` file

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
For fine-tuning we have changed only `--num_classes` to 25, `--dataset` to alpub, and `--use_imagenet_pretrained_weights` to `False`. Make sure to check file name, `hpo_search_space_config` file

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
For fine-tuning, we generated a `ckpt_path_dict` file during training and included it in the Baseline code for the fine-tuning process.

Here is an example of the bash code for fine-tuning the SimCLR model:
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

### `Without Pre-training` on Alpub:
#### Results on ResNet-18 without pretraining on Alpub dataset

We report the best-found augmentation and their corresponding validation and test set accuracies by directly fine-tuning on ICDAR. We observe that the baseline model achieves better results compared to the other two methods.

| **Experiment**    | **Dataset** | **Best Augmentation**                                         | **Valid Acc.** | **Test Acc.** |
|-------------------|-------------|---------------------------------------------------------------|----------------|---------------|
| Baseline model    | ICDAR       | randomcrop224,morpho_erosion,morpho_dilation,gaussianblur | 81.19%         | **80.67%**    |
| Triplet model     | ICDAR       | randomcrop224,morpho_dilation,affine,colorjitter          | 80.11%         | 79.16%        |
| SimCLR model      | ICDAR       | randomcrop224,affine,colorjitter,gray                     | 80.33%         | 80.00%        |

#### Results on ResNet-50 without pretraining on Alpub dataset

We report the best-found augmentation and their corresponding validation and test set accuracies. We observe that the baseline model achieves better results compared to the other two methods.

| **Experiment**    | **Dataset** | **Best Augmentation**                                  | **Valid Acc.** | **Test Acc.** |
|-------------------|-------------|-------------------------------------------------------|----------------|---------------|
| Baseline model    | ICDAR       | randomcrop224,morpho_erosion,gaussianblur          | 80.70%         | **80.47%**    |
| Triplet model     | ICDAR       | randomcrop224,morpho_erosion,morpho_dilation,gaussianblur | 79.29%         | 78.22%        |
| SimCLR model      | ICDAR       | randomcrop224,colorjitter,gaussianblur             | 80.05%         | 79.24%        |

### `With Pre-training` on Alpub:


Due to limited computational resources, we selected the top 4 augmentations for training on the Alpub dataset using two different strategies:

1. **Strategy 1: T-test based selection**  
   - `randomcrop198,morpho_dilation,hflip`
   - `randomcrop198,colorjitter,hflip,invert`
   - `randomcrop198,hflip,gray`
   - `randomcrop198,invert,gaussianblur,gray`

2. **Strategy 2: Best average validation accuracy**  
   - `randomcrop224,morpho_erosion,morpho_dilation,affine`
   - `randomcrop224,morpho_dilation,affine,colorjitter`
   - `randomcrop224,morpho_erosion,affine,colorjitter`
   - `randomcrop224,affine,colorjitter,gaussianblur`

These strategies helped us determine the most effective augmentations within our computational constraints.

#### Results on ResNet-18 with pretraining on Alpub dataset (with top-4 selected using strategy 1)

We report the best-found augmentation and their corresponding validation and test set accuracies. We observe that the baseline model achieves better results compared to the other two methods.

| **Experiment**    | **Dataset**          | **Best Augmentation**                              | **Valid Acc.** | **Test Acc.** |
|-------------------|----------------------|---------------------------------------------------|----------------|---------------|
| Baseline model    | `Alpub + ICDAR`         | randomcrop224,hflip,gray                      | 80.49%         | **79.94%**    |
| Triplet model     | `Alpub + ICDAR`       | randomcrop224,morpho_dilation,hflip           | 78.19%         | 77.51%        |
| SimCLR model      | `Alpub + ICDAR`         | randomcrop224,colorjitter,hflip,invert       | 77.55%         | 76.14%        |

#### Results on ResNet-50 with pretraining on Alpub dataset (with top-4 selected using strategy 1):
 We report the best found augmentation and their corresponding validation and test set accuracies. We observe the baseline model achieves the best results compared to the other two methods.

| **Experiment** | **Dataset**          | **Best augmentation**                               | **Valid Acc.** | **Test Acc.** |
|----------------|-----------------------|-----------------------------------------------------|----------------|---------------|
| Baseline model | `Alpub + ICDAR`       | randomcrop224,morpho_dilation,hflip             | 80.21%         | **79.75%**    |
| Triplet model  | `Alpub + ICDAR`       | randomcrop224,invert,gaussianblur,gray         | 77.90%         | 77.03%        |
| SimCLR model   | `Alpub + ICDAR`       | randomcrop224,invert,gaussianblur,gray         | 76.90%         | 76.59%        |



#### Results on ResNet-18 with pretraining on Alpub dataset (with top-4 selected using strategy 2):
 We report the best found augmentation and their corresponding validation and test set accuracies. We observe the baseline model achieves the best results compared to the other two methods.

| **Experiment** | **Dataset**          | **Best augmentation**                                 | **Valid Acc.** | **Test Acc.** |
|----------------|-----------------------|-------------------------------------------------------|----------------|---------------|
| Baseline model | `Alpub + ICDAR`       | randomcrop224,morpho_erosion,affine,colorjitter  | 80.68%         | **81.14%**    |
| Triplet model  | `Alpub + ICDAR`       | randomcrop224,morpho_dilation,affine,colorjitter | 79.57%         | 78.88%        |
| SimCLR model   | `Alpub + ICDAR`       | randomcrop224,morpho_erosion,affine,colorjitter  | 79.74%         | 79.18%        |


#### Results on ResNet-50 with pretraining on Alpub dataset (with top-4 selected using strategy 2):
 We report the best found augmentation and their corresponding validation and test set accuracies. We observe the baseline model achieves the best results compared to the other two methods.

| **Experiment** | **Dataset**          | **Best augmentation**                                 | **Valid Acc.** | **Test Acc.** |
|----------------|-----------------------|-------------------------------------------------------|----------------|---------------|
| Baseline model | `Alpub + ICDAR`       | randomcrop224,affine,colorjitter,gaussianblur    | 81.35%         | **81.17%**    |
| Triplet model  | `Alpub + ICDAR`       | randomcrop224,morpho_dilation,affine,colorjitter | 79.17%         | 78.24%        |
| SimCLR model   | `Alpub + ICDAR`       | randomcrop224,affine,colorjitter,gaussianblur    | 78.68%         | 78.85%        |


## t-SNE Analysis

To better understand this outcome, we provide embedding visualizations generated from different methods to support and explain these quantitative results. These visualizations will offer deeper insights into how the representations learned during pretraining might have affected the final model performance and why the expected improvements did not materialize.

### Comparison of t-SNE Visualizations of the Baseline Model
We visualize the embeddings of 1,000 data points from the ICDAR test set using the ResNet-18 backbone. The embeddings are derived from the feature representation just before the classification layer.


<table cellpadding="10" style="border: none;">
  <tr>
    <td align="center" style="border: none; padding-right: 20px;">
      <img src="Figures/ce_tsne_alpub_icdar_18_20_new.png" alt="Embeddings at the End of Pretraining on ALPUB" width="400"/>
      <p><strong>(a) Embeddings at the End of Pretraining on ALPUB</strong></p>
    </td>
    <td align="center" style="border: none; padding-left: 20px;">
      <img src="Figures/ce_tsne_alpub_18_seed2_20_finetune.png" alt="Embeddings at the End of Further Finetuning on ICDAR" width="400"/>
      <p><strong>(b) Embeddings at the End of Further Finetuning on ICDAR</strong></p>
    </td>
  </tr>
</table>


### Comparison of t-SNE Visualizations of the Triplet Model
The visualizations depict embeddings of 1,000 data points from the ICDAR test set using the ResNet-18 backbone. The pretraining and finetuning was conducted with the augmentations: `randomcrop224, invert, gaussianblur, gray`.

<table cellpadding="10">
  <tr>
    <td align="center" style="padding-right: 20px;">
      <img src="Figures/triplet_tsne_alpub_icdar_18_20_new.png" alt="Embeddings at the End of Pretraining on ALPUB" width="400" border="0"/>
      <p><strong>(a) Embeddings at the End of Pretraining on ALPUB</strong></p>
    </td>
    <td align="center" style="padding-left: 20px;">
      <img src="Figures/triplet_tsne_alpub_finetune_ce_20.png" alt="Embeddings at the End of Further Finetuning on ICDAR" width="400" border="0"/>
      <p><strong>(b) Embeddings at the End of Further Finetuning on ICDAR</strong></p>
    </td>
  </tr>
</table>


### Comparison of t-SNE Visualizations of the SimCLR Model
The visualizations depict embeddings of 1,000 data points from the ICDAR test set using the ResNet-18 backbone. The pretraining was conducted with the augmentations: `randomcrop224, hflip, gray`.

<table cellspacing="0" cellpadding="0">
  <tr>
    <td align="center" style="padding-right: 20px;">
      <img src="Figures/simclr_tsne_alpub_icdar_18_20_new.png" alt="Embeddings at the End of Pretraining on ALPUB" width="400" height="335"/>
      <p><strong>(a) Embeddings at the End of Pretraining on ALPUB</strong></p>
    </td>
    <td align="center" style="padding-left: 20px;">
      <img src="Figures/simclr_tsne_alpub_finetune_20.png" alt="Embeddings at the End of Further Finetuning on ICDAR" width="400" height="335"/>
      <p><strong>(b) Embeddings at the End of Further Finetuning on ICDAR</strong></p>
    </td>
  </tr>
</table>


Even though we expected pretraining on a larger and more diverse dataset to improve performance, the results didn’t show any significant gains. The cross-entropy Baseline model still delivered the best overall accuracy. This was backed up by t-SNE visualizations, which showed cleaner class separations in the Baseline model compared to both the Triplet and SimCLR models.


## Discussion, Limitations & Conclusion
### Discussion
#### SimCLR Cropping Scheme Leads to Semantic Shift in the Labels

<p align="center">
  <img src="Figures/simclr_letter.png" alt="SimCLR Cropping Scheme" width="400"/ />
</p>

For example, we observe the two views of the image cropped from the original image with 60% area. It can be seen that this cropping scheme leads to a change in the labels.

#### SimCLR Validation Loss

<p align="center">
  <img src="Figures/simclr_loss.png" alt="SimCLR Validation Loss" width="400"/ />
</p>

Comparison between ResNet-18 (left) and ResNet-50 (right) over 20 epochs.

To check if SimCLR training is converging, we analyze the validation loss across different augmentations and epochs (above figure). The decreasing loss indicates the model is improving. However, pretraining on the ALPUB dataset does not improve performance on downstream tasks, possibly due to errors introduced during the cropping phase that affected the model's generalization to new datasets.

### Limitations
**Hyperparameter Tuning**: We optimized data-augmentation strategies but could not exhaustively tune model-specific hyperparameters (e.g., dropout rates, optimizer parameters) due to hardware constraints. This may have affected model performance and generalizability.

**Data Augmentation**: We explored 10 data augmentation strategies, but the Albumentations library supports up to 40. Many were not tested, and fixed hyperparameters could have limited our insights into model performance.

**Cropping Size for SimCLR**: The choice to crop 60% of the original image was heuristic and not based on theory, which may have altered image semantics and affected performance.

**Batch Size in SimCLR**: SimCLR is sensitive to batch size. We used a reduced batch size of 115 instead of the typical 2048 due to computational limits, which might have impacted the model's ability to learn robust representations.

**Dataset Construction**: We used a fixed 70%-15%-15% split for training, validation, and testing. This could introduce bias, and multiple splits with averaged results may provide more reliable insights.

### Conclusion

This thesis evaluates SimCLR for Greek letter recognition and compares it with traditional models using cross-entropy and triplet loss. We tested 93 data augmentation techniques and found that SimCLR, despite its popularity, did not outperform traditional methods for this task.

The underperformance of SimCLR raises questions, particularly due to potential semantic shifts from cropping sub-images, which may affect letter recognition. Our analysis suggests that the baseline model with cross-entropy loss consistently performs better than both SimCLR and the triplet loss model.













