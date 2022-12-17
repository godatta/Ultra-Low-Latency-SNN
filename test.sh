python ann.py --dataset CIFAR10 --batch_size 256 --im_size 32 --architecture resnet18_multi_steps \
--learning_rate 1e-4 --epochs 1 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 \
--optimizer Adam --devices 0 --relu_threshold 1.0 \
--momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --linear_dropout 0 --conv_dropout 0 \
--hoyer_decay 0 --net_mode 'ori' --pool_pos 'before_relu' --test_type 'v1' \
--spike_type 'fixed' --bn_type 'bn' --start_spike_layer 0 --loss_type 'sum' --x_thr_scale 0.5 --weight_quantize 0 --use_hook --time_step 1 \
--pretrained_ann 'saved_models/ann_resnet18_multi_steps_cifar10_202212041432.pth' --get_layer_output
# --pretrained_ann 'saved_models/ann_resnet18_multi_steps_cifar10_202212041431.pth' --get_layer_output 
# --pretrained_ann 'saved_models/ann_resnet18_multi_steps_cifar10_202212041432.pth' --get_layer_output
#'/nas/home/zeyul/Knowledge-Distillation-Zoo/model_t/ann_vgg16_light_cifar10_202209290044.pth' 

# --get_layer_output
# 'trained_models_ann/ann_vgg16_relu_cifar10_202209212335.pth' --get_layer_output
# 'trained_models_ann/ann_vgg16_relu_cifar10_202209211209.pth' --get_layer_output
# 'trained_models_ann/ann_vgg16_relu_cifar10_202209201544.pth' 
# 'trained_models_ann/ann_vgg16_relu_cifar10_202209122042.pth'
# 'trained_models_ann/ann_vgg16_relu_cifar10_202209121539.pth' --get_layer_output
# '/nas/home/zeyul/mmdetection/checkpoints/resnet50_bn_spike_conv_0.8_SGD_best.pt'
# '/nas/home/zeyul/mmdetection/checkpoints/best.pt'

# torchrun --nproc_per_node=4 ann.py --dataset IMAGENET --batch_size 64 --im_size 224 --architecture resnet50 \
# --learning_rate 2e-1 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0005 --momentum 0.9 --amsgrad True --devices 0,1,2,3 --seed 0 --linear_dropout 0 --conv_dropout 0 \
# --net_mode 'ori' --pool_pos 'before_relu' --bn_type 'bn' --test_only \
# --spike_type 'sum' --loss_type 'sum' --hoyer_decay 2e-9 --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 --test_type 'v1' --use_hook \
# --pretrained_ann '/nas/home/zeyul/mmdetection/checkpoints/resnet50_bn_spike_conv_0.8_SGD_best.pt'
# Epoch: 1, best: 66.02%, test_loss: 1.4118, act_loss: 0.0000, total_test_loss: 1.4118, top1_acc: 66.02%, top5_acc: 77.20%, 
# output 0: 76.11%, relu: 0.00%, output threshold: 23.89%, time: 0:01:24
# 'trained_models_ann/ann_resnet20_cifar10_202208111842.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202207060058.pth'
# 93.92 
# 'trained_models_ann/ann_vgg16_cifar10_202206241620.pth'
# 93.82
# 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'
# 93.25
# 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth' 
# 93.15
# 'trained_models_ann/ann_vgg16_cifar10_202207062203.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206151759.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206131013.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206131841.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206072304.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206032334.pth'

# 'trained_models_ann/ann_vgg16_cifar10_202206151759.pth'
# 2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42, 44, 46 
# 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
# --conv_type 'dy'  --get_layer_output --get_scale 
# 202206131035 :
# 92.62 -> 92.63 -> 92.61 -> 92.65 -> 92.47 -> 92.32 -> 92.35 -> 91.73 --all--> 82.54


# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-4 --epochs 1 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 \
# --optimizer Adam --devices 0 --relu_threshold 4.0 \
# --momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --linear_dropout 0.3 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' \
# --act_mode 'mean' --bn_type 'fake' --start_spike_layer 0 --hoyer_type 'sum' --conv_type 'dy' \
# # --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206211113.pth'