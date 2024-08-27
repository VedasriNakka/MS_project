python -W ignore src/run.py -data ../datasets --dataset-name icdar \
         --log-every-n-steps 100 --epochs 100 --gpu-index 2 --lr 0.0003 --arch resnet18 \
         --transform_type resize96,hflip,colorjitter,gaussianblur --batch-size 16


# epochs=100
# augmentations=(

#     #"resize256,invert"\
#       "resize256"  #, "resize256,morpho_erosion", "resize256,morpho_dilation", "resize256,affine", 
# #     "resize256,colorjitter", "resize256,hflip", "resize256,invert", "resize256,gaussianblur",
# #     "resize256,morpho_erosion,morpho_dilation", "resize256,morpho_erosion,affine", 
# #     "resize256,morpho_erosion,colorjitter", "resize256,morpho_erosion,hflip", "resize256,morpho_erosion,invert",
# #     "resize256,morpho_erosion,gaussianblur", "resize256,morpho_dilation,affine", "resize256,morpho_dilation,colorjitter",
# #     "resize256,morpho_dilation,hflip", "resize256,morpho_dilation,invert", "resize256,morpho_dilation,gaussianblur",
# #     "resize256,affine,colorjitter", "resize256,affine,hflip", "resize256,affine,invert", "resize256,affine,gaussianblur", 
# #     "resize256,colorjitter,hflip", "resize256,colorjitter,invert", "resize256,colorjitter,gaussianblur", 
# #     "resize256,hflip,invert", "resize256,hflip,gaussianblur", "resize256,invert,gaussianblur",
# #     "resize256,morpho_erosion,morpho_dilation,affine", "resize256,morpho_erosion,morpho_dilation,colorjitter", 
# #     "resize256,morpho_erosion,morpho_dilation,hflip", "resize256,morpho_erosion,morpho_dilation,invert",
# #     "resize256,morpho_erosion,morpho_dilation,gaussianblur", "resize256,morpho_erosion,affine,colorjitter", 
# #     "resize256,morpho_erosion,affine,hflip", "resize256,morpho_erosion,affine,invert", 
# #     "resize256,morpho_erosion,affine,gaussianblur", "resize256,morpho_erosion,colorjitter,hflip",
# #     "resize256,morpho_erosion,colorjitter,invert", "resize256,morpho_erosion,colorjitter,gaussianblur",
# #     "resize256,morpho_erosion,hflip,invert", "resize256,morpho_erosion,hflip,gaussianblur", 
# #     "resize256,morpho_erosion,invert,gaussianblur", "resize256,morpho_dilation,affine,colorjitter", 
# #     "resize256,morpho_dilation,affine,hflip", "resize256,morpho_dilation,affine,invert", 
# #     "resize256,morpho_dilation,affine,gaussianblur", "resize256,morpho_dilation,colorjitter,hflip", 
# #     "resize256,morpho_dilation,colorjitter,invert", "resize256,morpho_dilation,colorjitter,gaussianblur",
# #     "resize256,morpho_dilation,hflip,invert", "resize256,morpho_dilation,hflip,gaussianblur",
# #     "resize256,morpho_dilation,invert,gaussianblur", "resize256,affine,colorjitter,hflip", 
# #     "resize256,affine,colorjitter,invert", "resize256,affine,colorjitter,gaussianblur",
# #     "resize256,affine,hflip,invert", "resize256,affine,hflip,gaussianblur", 
# #     "resize256,affine,invert,gaussianblur", "resize256,colorjitter,hflip,invert",
# #     "resize256,colorjitter,hflip,gaussianblur", "resize256,colorjitter,invert,gaussianblur", 
# #     "resize256,hflip,invert,gaussianblur"

    

# )

# ##Iterate over each augmentation combination
# for augmentation in "${augmentations[@]}"; do
    
# CUDA_VISIBLE_DEVICES=3 python src/run.py \
#         -data ../datasets \
#         -dataset-name HomerCompTraining_Cropped \
#         --log-every-n-steps 100 \
#         --arch resnet18 \
#         --batch-size 16  \
#         --epochs ${epochs} \
#         --transform_type ${augmentation} \
#         #--transform_type randomcrop


# done