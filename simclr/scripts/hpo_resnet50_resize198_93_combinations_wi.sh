SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Script path: $SCRIPT_PATH"


epochs=100


dataset=icdar

#hpo_exp_name="resize64_65_combinations"
hpo_exp_name="${dataset}_50_resize198_93_com_resnet50_wi"
log_filepath=${hpo_exp_name}_$(date +'%Y-%m-%d_%H-%M-%S').log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v2.py \
        --dataset_name ${dataset} \
        --num_classes 25 \
        --arch resnet50 \
        --batch_size 115  \
        --epochs ${epochs} \
        --log-every-n-steps 100 \
        --gpu-index 2 \
        --lr 0.0003 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --gpu_per_worker 1.0 \
        --cpu_per_worker 2 \
        --out_dim  128 \
        --launch_script_path ${SCRIPT_PATH} \
        --hpo_search_space_config "scripts/search_space/search_space_resize198_93_combinations.yaml" $@ >& ./logs/${log_filepath}
        
        