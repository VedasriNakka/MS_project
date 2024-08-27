
epochs=20
hpo_exp_name="CE_loss_full_sweep_$(date +'%Y-%m-%d_%H-%M-%S')"
log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v1.py \
        --mode finetune \
        --arch resnet18 \
        --batch_size 64  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 0.33 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn ce \
        --triplet_embedding_size 64 \
        --hpo_search_space_config "scripts/search_space/search_space_double.yaml" $@ >& ./logs/${log_filepath} 




