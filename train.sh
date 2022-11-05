
# train VGG16
python ann.py --dataset CIFAR10 --batch_size 128 --im_size 32 --architecture vgg16_light \
--learning_rate 1e-4 --epochs 600 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer Adam --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
--hoyer_decay 1e-8 --net_mode 'ori' --pool_pos 'before_relu'  --if_set_0 \
--spike_type 'cw' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
--description 'train a spike vgg16' --warmup 0 --lr_decay 'step' --reg_thr  --use_wandb --use_hook --log

# train ResNet20 
python ann.py --dataset CIFAR10 --batch_size 128 --im_size 32 --architecture resnet18 \
--learning_rate 1e-1 --epochs 600 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0 --conv_dropout 0 \
--hoyer_decay 1e-8 --net_mode 'ori' --pool_pos 'before_relu' \
--spike_type 'cw' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
--description 'train a spike resnet18' --warmup 0 --lr_decay 'step' --reg_thr  --use_wandb --use_hook --log


