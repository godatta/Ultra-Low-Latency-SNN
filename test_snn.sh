# test snn
python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 1 \
--lr_interval '0.60 0.80 0.90' --lr_reduce 1 --timesteps 1 --leak 1.0 --scaling_factor 1.0 \
--optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 100 --devices 0 --default_threshold 1.0 \
--test_only --dont_save \
--pretrained_snn 'trained_snn_models/snn_vgg16_cifar10_1_202205232213.pth'
# 'trained_snn_models/snn_vgg16_cifar10_1_202204120001_91.33.pth' 
# 'trained_snn_models/snn_vgg16_cifar10_1_202204091115/epoch_120_0.9109.pth'
# 'trained_snn_models/snn_vgg16_cifar10_1_202204071241/epoch_360.9089.pth'
# 'trained_snn_models/snn_vgg16_cifar10_1_202204091115/epoch_120_0.9109.pth'
# 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'
# --layer_output
# --pretrained_snn 'trained_snn_models/snn_vgg16_cifar10_1test002.pth' --test_only --dont_save 89.89
# 'trained_snn_models/snn_vgg16_cifar10_1_202204010923/epoch_107.pth' 90.11

# 'trained_snn_models/snn_vgg16_cifar10_1_202204061143/epoch_36.pth'

# 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'
