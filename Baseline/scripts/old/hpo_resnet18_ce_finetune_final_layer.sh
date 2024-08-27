
epochs=20
hpo_exp_name="CE_with_simclr_without_backbone_93_combinations_$(date +'%Y-%m-%d_%H-%M-%S')"
#hpo_exp_name="TEMP_$(date +'%Y-%m-%d_%H-%M-%S')"

log_filepath=${hpo_exp_name}.log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore  src/hpo_v1.py \
        --mode finetune_simclr \
        --freeze_backbone True \
        --arch resnet18 \
        --batch_size 64  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 0.45 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --loss_fn ce \
        --triplet_embedding_size 64 \
        --ckpt_path_dict /home/vedasri/SimCLR/results_hpo/InfoNCE_93_combinations_v1_2024-06-10_09-39-25/config2ckpt_path.json \
        --hpo_search_space_config "scripts/search_space/search_space_finetune_with_simclr.yaml" $@ >& ./logs/${log_filepath} 




