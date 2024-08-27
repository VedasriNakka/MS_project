
epochs=2


# hpo_exp_name="single_augmentation"
# log_filepath=${hpo_exp_name}_$(date +'%Y-%m-%d_%H-%M-%S').log

    
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo.py \
#         --mode finetune \
#         --arch resnet18 \
#         --batch_size 64  \
#         --epochs ${epochs} \
#         --dataset icdar \
#         --gpu_per_worker 0.5 \
#         --hpo_exp_name ${hpo_exp_name} \
#         --log_filepath ${log_filepath} \
#         --hpo_search_space_config "scripts/search_space/search_space_single.yaml" $@ >& ./logs/${log_filepath} 






#hpo_exp_name="gray_augmentation_20epochs"
hpo_exp_name="all_augmentations_20epochs_seed5"
log_filepath=${hpo_exp_name}_$(date +'%Y-%m-%d_%H-%M-%S').log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo.py \
        --mode finetune \
        --arch resnet18 \
        --batch_size 64  \
        --epochs ${epochs} \
        --dataset icdar \
        --gpu_per_worker 0.5 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --hpo_search_space_config "scripts/search_space/search_space_double.yaml" $@ >& ./logs/${log_filepath} 




