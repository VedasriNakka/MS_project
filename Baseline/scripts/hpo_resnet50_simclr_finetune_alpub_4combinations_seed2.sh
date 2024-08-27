SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=20
hpo_exp_name="alpub_resnet50_SimCLR_finetine_with_ce_loss_unfreeze_backbone_4_agumentations_seed2_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
        --mode simclr_finetune_with_ce \
        --num_classes 25 \
        --arch resnet50 \
        --batch_size 80  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 1.0 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn simclr_finetune_with_ce \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights False \
        --freeze_backbone False \
        --ckpt_path_dict /home/vedasri/SimCLR_V2/results_hpo/final_experiments/alpub_alpub_resize198_4_combinations_100epochs_resnet50_with_seed2/config2ckpt_path.json \
        --hpo_search_space_config "/home/vedasri/Baseline_V2/scripts/search_space/search_space_4_combinations_alpub_simclr_seed2.yaml" $@ >& ./logs/${log_filepath} 




# SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# echo "Script path: $SCRIPT_PATH"


# epochs=20
# hpo_exp_name="alpub_resnet50_SimCLR_finetine_with_ce_loss_freeze_backbone_4_agumentations_seed2_$(date +'%Y-%m-%d_%H-%M-%S')"
# log_filepath=${hpo_exp_name}.log

    
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
#         --mode simclr_finetune_with_ce \
#         --arch resnet50 \
#         --batch_size 128  \
#         --epochs ${epochs} \
#         --dataset icdar \
#         --gpu_per_worker 1.0 \
#         --hpo_exp_name ${hpo_exp_name} \
#         --log_filepath ${log_filepath} \
#         --loss_fn simclr_finetune_with_ce \
#         --launch_script_path ${SCRIPT_PATH} \
#         --use_imagenet_pretrained_weights False \
#         --freeze_backbone True \
#         --ckpt_path_dict /home/vedasri/SimCLR_V2/results_hpo/final_experiments/alpub_alpub_resize198_4_combinations_100epochs_resnet50_with_seed2/config2ckpt_path.json \
#         --hpo_search_space_config "/home/vedasri/Baseline_V2/scripts/search_space/search_space_4_combinations_alpub_simclr_seed2.yaml" $@ >& ./logs/${log_filepath} 


