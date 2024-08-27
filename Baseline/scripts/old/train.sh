
epochs=50

#CUDA_VISIBLE_DEVICES=2 python src/train.py --mode finetune --arch resnet18 --batch_size 64 --epochs ${epochs} --transform_type center
#CUDA_VISIBLE_DEVICES=2 python src/train.py --mode finetune --arch resnet34 --batch_size 64 --epochs ${epochs} --transform_type center
CUDA_VISIBLE_DEVICES=0 python src/train.py --mode finetune --arch resnet50 --batch_size 64 --epochs ${epochs} --transform_type center
