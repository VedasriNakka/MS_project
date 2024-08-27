epochs=100


hpo_exp_name="InfoNCE"
log_filepath=${hpo_exp_name}_$(date +'%Y-%m-%d_%H-%M-%S').log

    
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/hpo_v2.py \
        --dataset_name icdar \
        --arch resnet18 \
        --batch_size 256  \
        --epochs ${epochs} \
        --log-every-n-steps 100 \
        --gpu-index 2 \
        --lr 0.0003 \
        --hpo_exp_name ${hpo_exp_name} \
        --log_filepath ${log_filepath} \
        --hpo_search_space_config "scripts/search_space/search_space_simple.yaml" $@ >& ./logs/${log_filepath}
        
        
        ##--dataset-name HomerCompTraining_Cropped



        # src/run.py -data ../datasets --dataset-name HomerCompTraining_Cropped \
        #  --log-every-n-steps 100 --epochs 100 --gpu-index 2 --lr 0.0003 --arch resnet18 \
        #  --transform_type resize96,hflip,colorjitter,gaussianblur --batch-size 16