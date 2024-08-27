
epochs=50


CUDA_VISIBLE_DEVICES=1 python src/train.py --mode finetune --arch resnet50 --batch_size 64 --epochs ${epochs} --transform_type full
