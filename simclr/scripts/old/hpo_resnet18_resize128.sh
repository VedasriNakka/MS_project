epochs=150


#hpo_exp_name="resize64_65_combinations"
hpo_exp_name="resize64_resize128_4_combinations"
log_filepath=${hpo_exp_name}_$(date +'%Y-%m-%d_%H-%M-%S').log

    
CUDA_VISIBLE_DEVICES=0 python src/hpo_v2.py \
        --dataset_name alpub \
        --num_classes 24 \
        --arch resnet18 \
        --batch_size 256  \
        --epochs ${epochs} \
        --log-every-n-steps 100 \
        --gpu-index 2 \
        --lr 0.0003 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --gpu_per_worker 1.0 \
        --cpu_per_worker 2 \
        --hpo_search_space_config "scripts/search_space/search_space_64_resize128.yaml" $@ >& ./logs/${log_filepath}
        
        