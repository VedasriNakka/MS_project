SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=20
hpo_exp_name="CE_loss_resnet50_4_augmentations_alpub_icdar_seed2_finetune_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
        --mode finetune_with_ce \
        --arch resnet50 \
        --num_classes 25 \
        --batch_size 78  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 1.0 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn ce \
        --triplet_embedding_size 64 \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights False \
        --freeze_backbone False \
        --ckpt_path_dict /home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/CE_loss_resnet50_4_augmentations_alpub_seed2_2024-08-11_14-47-45/config2ckpt_path.json \
        --hpo_search_space_config "scripts/search_space/search_space_4_combinations_alpub_ce_seed2.yaml" $@ >& ./logs/${log_filepath} 




