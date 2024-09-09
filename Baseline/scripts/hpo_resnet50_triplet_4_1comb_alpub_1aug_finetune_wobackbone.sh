SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"

epochs=20
hpo_exp_name="Triplet_resnet50_alpub_with_ce_loss_freeze_backbone_4_1_aug_finetune_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=1 python src/hpo_v1.py \
        --mode triplet_finetune_with_ce \
        --arch resnet50 \
        --batch_size 64  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 1.0 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn triplet_finetune_with_ce \
        --triplet_embedding_size 64 \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights False \
        --freeze_backbone True \
        --ckpt_path_dict //home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_resnet50_loss_alpub_4_1_agumentations_seed_1aug_new_2024-09-05_17-45-05/config2ckpt_path.json \
        --hpo_search_space_config "scripts/search_space/search_space_4_combinations_alpub_triplet_seed2_1aug.yaml" $@ >& ./logs/${log_filepath} 

