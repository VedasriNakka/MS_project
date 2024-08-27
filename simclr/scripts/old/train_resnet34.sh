python -W ignore src/run.py -data ../datasets -dataset-name HomerCompTraining_Cropped \
        --log-every-n-steps 100 --epochs 100 --gpu-index 1 --lr 0.0003 --arch resnet34 --transform_type randomcrop
