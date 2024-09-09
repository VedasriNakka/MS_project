SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=20
hpo_exp_name="Triplet_resnet50_loss_alpub_4_1_agumentations_seed_1aug_new_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=3 python src/hpo_v1.py \
        --mode finetune \
        --arch resnet50 \
        --num_classes 24 \
        --batch_size 14  \
        --epochs ${epochs} \
        --dataset alpub \
        --gpu_per_worker 1.0 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn triplet \
        --triplet_embedding_size 64 \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights True \
        --hpo_search_space_config "scripts/search_space/search_space_4_combinations_alpub_triplet_seed2_1aug.yaml" $@ >& ./logs/${log_filepath} 




