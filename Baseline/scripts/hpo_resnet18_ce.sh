SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=2
hpo_exp_name="CE_loss_demo_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0 python src/hpo_v1.py \
        --mode finetune \
        --arch resnet18 \
        --batch_size 64  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 1.0 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn ce \
        --triplet_embedding_size 64 \
        --launch_script_path ${SCRIPT_PATH} \
        --use_imagenet_pretrained_weights True \
        --hpo_search_space_config "scripts/search_space/search_space_subset_2.yaml" $@ >& ./logs/${log_filepath} 




