SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=20
hpo_exp_name="SimCLR_finetine_with_ce_loss_unfreeze_backbone_93_agumentations_seed2_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
        --mode simclr_finetune_with_ce \
        --arch resnet18 \
        --batch_size 64  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 0.5 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn simclr_finetune_with_ce \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights False \
        --freeze_backbone False \
        --ckpt_path_dict /home/vedasri/SimCLR_V2/results_hpo/icdar_simclr_resnet18_resize198_93_combinations_100epochs_seed2_186_2024-07-31_18-20-11/config2ckpt_path.json \
        --hpo_search_space_config "scripts/search_space/search_space_subset_93_seed2.yaml" $@ >& ./logs/${log_filepath} 




# SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# echo "Script path: $SCRIPT_PATH"


# epochs=20
# hpo_exp_name="SimCLR_finetine_with_ce_loss_freeze_backbone_93_agumentations_seed2_$(date +'%Y-%m-%d_%H-%M-%S')"
# log_filepath=${hpo_exp_name}.log

    
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
#         --mode simclr_finetune_with_ce \
#         --arch resnet18 \
#         --batch_size 64  \
#         --epochs ${epochs} \
#         --dataset icdar \
#         --gpu_per_worker 0.5 \
#         --hpo_exp_name ${hpo_exp_name} \
#         --log_filepath ${log_filepath} \
#         --loss_fn simclr_finetune_with_ce \
#         --launch_script_path ${SCRIPT_PATH} \
#         --use_imagenet_pretrained_weights False \
#         --freeze_backbone True \
#         --ckpt_path_dict /home/vedasri/SimCLR_V2/results_hpo/icdar_resize198_93_combinations_infonce_loss_100epochs_2024-07-05_17-51-49/config2ckpt_path.json \
#         --hpo_search_space_config "scripts/search_space/search_space_subset_93_seed2.yaml" $@ >& ./logs/${log_filepath} 


