from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle

output_file = 'network_output/ann_vgg16_cifar10_202206201914_test'
# 'network_output/ann_vgg16_cifar10_202206091914_test'
# 'network_output/ann_vgg16_cifar10_202205182136_test'
# 'network_output/snn_vgg16_cifar10_1_202205221408_test'
# 'network_output/ann_vgg16_cifar10_4.0_0.2lr_decay_test'
# 'network_output/ann_vgg16_cifar10_202205192323_test'


# model = torch.load(os.path.join('trained_models_ann', output_file.split('/')[-1][:-5])+'.pth')
def plot(mode = 'ann'):
    threshold = {}
    update_thr = {}
    if mode == 'ann':
        model = torch.load(os.path.join('trained_models_ann', output_file.split('/')[-1][:-5])+'.pth', map_location='cpu')
        # with open('new_factor_scale/new_factor_x_vgg16_cifar10_1','rb') as f:
        #     scale_factor = pickle.load(f)
    else:
        model = torch.load(os.path.join('trained_snn_models', output_file.split('/')[-1][:-5])+'.pth', map_location='cpu')
        with open('new_factor_scale/new_factor_x_vgg16_cifar10_1','rb') as f:
            scale_factor = pickle.load(f)
        with open('output/ann_max_vgg16_cifar10_0.9', 'rb') as f:
            max_thr_scale = torch.load(f, map_location='cpu')

    for key in model['state_dict'].keys():
        if key[:9] == 'threshold':
            threshold[key[11:]] = model['state_dict'][key].cpu()
    if mode == 'snn':
        for i,key in enumerate(sorted(threshold.keys(), key=lambda k: int(k))):
            update_thr[key] = threshold[key]*scale_factor[i]
            print('key: {}, before: {}, update: {}'.format(key, threshold[key], update_thr[key]))
    
    output = torch.load(output_file)
    plt.figure(figsize=(32,32))
    min = max = 0.0
    mid_hoyers = []
    for i,k in enumerate(output.keys()):
        print('k: {}'.format(k))
        plt.subplot(4,4,i+1)
        nums = np.asarray(output[k])
        nums_batch = nums.reshape(128,-1)
        hoyer_line = np.sum(nums_batch**2) / np.sum(np.abs(nums_batch)) if np.sum(np.abs(nums_batch)) > 0 else 0.0
        # (np.linalg.norm(nums,2)**2/np.linalg.norm(nums,1))
        
        hoyer_sum = np.mean(np.sum(nums_batch**2, axis=1) / np.sum(np.abs(nums_batch), axis=1)) if np.sum(np.abs(nums_batch)) > 0 else 0.0
        nums_max = np.max(nums)
        
        if k != 'total':
            total = nums.shape[0]
            layer_min = (nums<=0).sum() / total * 100.0
            layer_max = (nums>=nums_max).sum()  / total * 100.0
            # layer_max = (nums>=threshold[str(k)].item()).sum()  / total * 100.0
            min += layer_min
            max += layer_max
            if mode == 'snn':
                plt.vlines(update_thr[str(k)].item(), 0, 10e4, linestyles='dashed', color='g', label='updated_thr')
                plt.vlines(max_thr_scale[i], 0, 10e4, linestyles='dashed', color='b', label='max_thr_scale')
            plt.hist(output[k], label='{}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k, layer_min, 100-layer_min-layer_max, layer_max), bins=100)
            plt.vlines(nums_max, 0, 10e4, linestyles='dashed', color='b', label='threshold')
            # plt.vlines(threshold[str(k)].item() * scale_factor[i], 0, 10e4, linestyles='dashed', color='g', label='updated_threshold')
            # plt.vlines(threshold[str(k)].item(), 0, 1e4, linestyles='dashed', color='b', label='threshold')
            plt.vlines(hoyer_line, 0, 10e4, linestyles='dashed',color='r', label='hoyer_line')
            plt.vlines(hoyer_sum, 0, 10e5, linestyles='dotted',color='g', label='hoyer_line_sum')

            mid_num = nums[nums>0.0]
            mid_num = mid_num[mid_num<nums_max]
            mid_hoyer = np.sum(mid_num**2) / np.sum(np.abs(mid_num)) if  np.sum(np.abs(mid_num)) > 0 else 0.0
            mid_hoyers.append(mid_hoyer/nums_max)
            # mid_num = torch.tensor(mid_num)
            # mid_total = mid_num.shape[0]
            # min_scale = mid_num.kthvalue(int(mid_total*0.2)).values.item()
            # max_scale = mid_num.kthvalue(int(mid_total*0.8)).values.item()

            # plt.vlines(min_scale, 0, 10e6, linestyles='dashed', label='min_scale')
            # plt.vlines(max_scale, 0, 10e6, linestyles='dotted', label='max_scale')
            plt.vlines(mid_hoyer, 0, 10e7, linestyles='dotted', label='mid_hoyer')
        else:
            min /= (len(output.keys())-1)
            max /= (len(output.keys())-1)
            plt.hist(output[k], label='{}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k, min, 100-min-max, max), bins=100)
        plt.legend()
        plt.yscale('log')
    plt.savefig(output_file[:-4] + '.jpg')
    # torch.save(mid_hoyers, 'network_output/my_x_scale_factor_1753')

def plot_hist():  
    output = torch.load(output_file)
    plt.figure(figsize=(32,32))
    min = max = 0.0
    mid_hoyers = []
    # bins_num = [2**(32*32), 2**(32*32), 2**(16*16), 2**(16*16), 2**(8*8), 2**(8*8), 2**(8*8), 2**(4*4), 2**(4*4), 2**(4*4), 2**(2*2), 2**(2*2), 2**(2*2)]

    for i,k in enumerate(output.keys()):
        print('k: {}, shape: {}'.format(k, output[k].shape))
        plt.subplot(4,4,i+1)
        nums = np.asarray(output[k])
        
        if k != 'total':
            plt.hist(output[k], label='{}'.format(k), bins=1024)
            plt.legend()
            plt.yscale('log')
    plt.savefig(output_file[:-4] + '_v2.jpg')
    # torch.save(mid_hoyers, 'network_output/my_x_scale_factor_1753')

if __name__ == '__main__':
    # plot_hist()
    plot()
    # plot('snn')
