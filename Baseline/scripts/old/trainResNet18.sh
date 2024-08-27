
epochs=50

# CUDA_VISIBLE_DEVICES=2 python src/train.py --mode finetune --arch resnet18 --batch_size 64  \
#         --epochs ${epochs} --transform_type thinning \
#         --dataset alpub 


# CUDA_VISIBLE_DEVICES=2 python src/train.py --mode finetune --arch resnet18 --batch_size 64  \
#         --epochs ${epochs} --transform_type "default" \
#         --dataset icdar 




epochs=50
# for augmentations in   
#         #"resize256" \
#          "resize256,hflip" \
#         "resize256,hflip,colorjitter" \
#         "resize256,hflip,morpho_dilation" \
#         "resize256,hflip,morpho_erosion"  \
#         "resize256,hflip,affine" \
#         "resize256,hflip,affine";

# Define the list of augmentation combinations
augmentations=(

    #"resize256,invert"\
    "resize256,hflip,invert"\
    "resize256,affine,invert"\
    "resize256,affine,invert,colorjitter"\
    "resize256,affine,invert,morpho_erosion"\
    "resize256,affine,invert,morpho_dilation"\
    "resize256,hflip,affine,invert,colorjitter"\
    "resize256,hflip,affine,invert,morpho_erosion"\
    "resize256,hflip,affine,invert,morpho_dilation"\
    
    # "resize256,morpho_erosion"\
    # "resize256,morpho_dilation"\
    # "resize256,affine" \
    # "resize256,colorjitter" \
    # "resize256,affine,colorjitter"\
    # "resize256" \
    #"resize256,hflip" \
    #"resize256,hflip,colorjitter" \
    #"resize256,hflip,morpho_dilation" \
    #"resize256,hflip,morpho_erosion"  \
)

# Iterate over each augmentation combination
for augmentation in "${augmentations[@]}"; do
    
CUDA_VISIBLE_DEVICES=2 python src/train.py \
        --mode finetune \
        --arch resnet18 \
        --batch_size 8  \
        --epochs ${epochs} \
        --transform_type ${augmentation} \
        --dataset icdar 

done
