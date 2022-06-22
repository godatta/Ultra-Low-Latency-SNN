python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
--learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 1 --leak 1.0 --scaling_factor 0.5 \
--optimizer Adam --weight_decay 0.0005 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 1000 --devices 0 --default_threshold 1.0 --log \
--decay 0 --reg_type 0 --act_decay 1e-5 \
--description 'use new x_factor' \
--pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202205291753.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202205182136.pth'
# 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202205171206.pth'
# --pretrained_snn 'trained_snn_models/snn_vgg16_cifar10_1_202204120001_91.33.pth'
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'