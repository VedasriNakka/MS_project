SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=6
hpo_exp_name="Triplet_loss_2_agumentations_demo_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
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
        --hpo_search_space_config "scripts/search_space/search_space_2_combinations_triplet_demo.yaml" $@ >& ./logs/${log_filepath} 




