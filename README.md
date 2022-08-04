# pretrained model:
test_acc: 0.9325, output 0: 88.10%, relu: 8.60%, output threshold: 3.30%,
path: trained_models_ann/ann_vgg16_cifar10_202206122147.pth
https://drive.google.com/file/d/179VAvRHnRswKrobXuw61gZAAYU5X5uip/view?usp=sharing

# Train

For ImageNet:
``` bash
bash tdbn_dist_train.sh
```
In tdbn_dist_train.sh:
``` bash
torchrun --nproc_per_node=4 ann.py --devices 0,1,2,3 --dataset IMAGENET --batch_size 32 --im_size 224 --architecture VGG16 \
--learning_rate 1e-1 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True  --seed 0 --linear_dropout 0.3 --conv_dropout 0.3 \
--net_mode 'ori' --log --pool_pos 'before_relu'  --bn_type 'bn' --warmup \
--act_mode 'fixed' --hoyer_type 'sum' --hoyer_decay 1e-8 --start_spike_layer 50 --x_thr_scale 1.0 --weight_quantize 0 \
--description 'imagenet test without wandb'
```

- nproc_per_node: the num of gpus
- devices: 0,1,2,3
- net_mode: 'ori': original model; 'cut_n': when x < n*10%, x=0, when x > n*10%, x=1.0
- pool_pos: the position of pooling layer, 'before_relu' and 'after_relu'
- bn_type: the type of batchnorm layers:
    - bn: original bn
    - fake: original bn * a fixed thr
    - tdbn: original bn * the thr of corresponding layer
- act_mode: 
    - fixed: threshold always is 1.0, 
    - sum: use sum hoyer as thr, 
    - channelwise(cw): use cw hoyer as thr 
- hoyer_type: the type of hoyer regularization, 'sum', 'cw', 'mean'
- hoyer_decay: the coefficient of hoyer regularization 
- start_spike_layer: 
    - 0: all layers would be spiked
    - 50: no layer would be spiked
- x_thr_scale: scale the thrshold, for an all-spiked vgg16 in CIFAR10, 0.618 is good
- weight_quantize: the num of bit to quantize the weights



