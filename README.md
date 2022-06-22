# pretrained model:
test_acc: 0.9325, output 0: 88.10%, relu: 8.60%, output threshold: 3.30%,
path: trained_models_ann/ann_vgg16_cifar10_202206122147.pth
https://drive.google.com/file/d/179VAvRHnRswKrobXuw61gZAAYU5X5uip/view?usp=sharing

# Train
Because the fixed threshold does not work good, 

I update it by calculating the scales (x>min_thr -> x>thr*min_scale),


I updated the source code, and the command line is same.

Exp1: bash tdbn_train.sh

Exp2: change the ``` --net_mode 'cut_2' ``` to ``` --net_mode 'cut_3' ```, and run bash tdbn_train.sh



