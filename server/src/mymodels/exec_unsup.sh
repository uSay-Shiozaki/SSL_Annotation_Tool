python3 -m torch.distributed.launch --nproc_per_node=1 ./mycodes/for_ibot/my_unsup_cls.py \
--arch vit_large \
--checkpoint_key student \
--pretrained_weights './ibot_large_pretrain.pth' \
--data_path "/mnt/media/irielab/win_drive/ImageNet/imagenet-object-localization-challenge-2012/ILSVRC/Data/CLS-LOC/10class_train_val" \