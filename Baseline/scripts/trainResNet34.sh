
epochs=50



UDA_VISIBLE_DEVICES=3 python src/train.py --mode finetune --arch resnet34 --batch_size 64 --epochs ${epochs} --transform_type full
