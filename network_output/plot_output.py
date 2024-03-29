from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle
import seaborn as sns

output_file = 'network_output/ann_resnet18_multi_steps_cifar10_202212041432_test'
output_file2 = 'network_output/ann_resnet18_multi_steps_cifar10_202212041431_test'
# 'network_output/ann_resnet18_multi_steps_cifar10_202212041432_test'
# 'network_output/ann_vgg16_relu_cifar10_202209122042_test'
# 'network_output/ann_vgg16_relu_cifar10_202209121539_test'
# 'network_output/ann_vgg16_cifar10_202206091914_test'
# 'network_output/ann_vgg16_cifar10_202205182136_test'
# 'network_output/snn_vgg16_cifar10_1_202205221408_test'
# 'network_output/ann_vgg16_cifar10_4.0_0.2lr_decay_test'
# 'network_output/ann_vgg16_cifar10_202205192323_test'


# model = torch.load(os.path.join('trained_models_ann', output_file.split('/')[-1][:-5])+'.pth')
def subplot():
    
    output = torch.load(output_file)
    plt.figure(figsize=(32,32))
    min = max = 0.0
    count = 0
    for i,k in enumerate(output.keys()):
        if k not in [1,10,11,12]:
            continue
        count += 1
        print('k: {}, type: {}, shape: {}'.format(k,type(k), output[k].shape))
        plt.subplot(2,2,count)
        nums = np.asarray(output[k])
        nums_max = np.max(nums)
        
        if k in [1,10,11,12]:
            total = nums.shape[0]
            layer_min = (nums<=0).sum() / total * 100.0
            layer_max = (nums>=nums_max).sum()  / total * 100.0
            min += layer_min
            max += layer_max
        
            plt.hist(output[k], bins=128)

            # plt.legend()
            plt.yscale('log')
    plt.savefig(output_file[:-4] + '_sub.jpg')
    # torch.save(mid_hoyers, 'network_output/my_x_scale_factor_1753')

def combine_plot(mode = 'ann'):
    threshold = {}
    update_thr = {}
    if mode == 'ann':
        pass
        # model = torch.load(os.path.join('trained_models_ann', output_file.split('/')[-1][:-5])+'.pth', map_location='cpu')
        # with open('new_factor_scale/new_factor_x_vgg16_cifar10_1','rb') as f:
        #     scale_factor = pickle.load(f)
    else:
        model = torch.load(os.path.join('trained_snn_models', output_file.split('/')[-1][:-5])+'.pth', map_location='cpu')
        with open('new_factor_scale/new_factor_x_vgg16_cifar10_1','rb') as f:
            scale_factor = pickle.load(f)
        with open('output/ann_max_vgg16_cifar10_0.9', 'rb') as f:
            max_thr_scale = torch.load(f, map_location='cpu')

    # for key in model['state_dict'].keys():
    #     if key[:9] == 'threshold':
    #         threshold[key[11:]] = model['state_dict'][key].cpu()
    if mode == 'snn':
        for i,key in enumerate(sorted(threshold.keys(), key=lambda k: int(k))):
            update_thr[key] = threshold[key]*scale_factor[i]
            print('key: {}, before: {}, update: {}'.format(key, threshold[key], update_thr[key]))
    
    output = torch.load(output_file)
    output2 = torch.load(output_file2)
    plt.figure(figsize=(32,32))
    min = max = 0.0
    mid_hoyers = []
    for i,k in enumerate(output.keys()):
        print('k: {}, shape: {}, shape2: {}'.format(k, output[k].shape, output2[k].shape))
        plt.subplot(5,5,i+1)
        nums = np.asarray(output[k])
        nums2 = np.asarray(output2[k])
        # nums_batch = nums.reshape(128,-1)
        # hoyer_line = np.sum(nums_batch**2) / np.sum(np.abs(nums_batch)) if np.sum(np.abs(nums_batch)) > 0 else 0.0
        # (np.linalg.norm(nums,2)**2/np.linalg.norm(nums,1))
        
        # hoyer_sum = np.mean(np.sum(nums_batch**2, axis=1) / np.sum(np.abs(nums_batch), axis=1)) if np.sum(np.abs(nums_batch)) > 0 else 0.0
        nums_max = np.max(nums)
        nums2_max = np.max(nums2)
        
        if k != 'total':
        # if k in [1,10,11,12]:
        # if k == 1:
            total = nums.shape[0]
            layer_min = (nums<=0).sum() / total * 100.0
            layer_max = (nums>=0.5).sum()  / total * 100.0
            layer_min2 = (nums2<=0).sum() / total * 100.0
            layer_max2 = (nums2>=1.0).sum()  / total * 100.0
            # layer_max = (nums>=threshold[str(k)].item()).sum()  / total * 100.0
            # min += layer_min
            # max += layer_max
            alpha = 1.0
            color1 = '#1f77b4'
            color2 = '#ff7f0e'
            if mode == 'snn':
                plt.vlines(update_thr[str(k)].item(), 0, 10e4, linestyles='dashed', color='g', label='updated_thr')
                plt.vlines(max_thr_scale[i], 0, 10e4, linestyles='dashed', color=color2, label='max_thr_scale')
            if nums_max < nums2_max:
                plt.hist(output2[k], label='layer {}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k+1, layer_min2, 100-layer_min2-layer_max2, layer_max2), bins=100, color=color1, alpha=alpha)
                plt.vlines(nums2_max, 0, 10e4, linestyles='dashed', color=color1, label='threshold 1', alpha=alpha)
                plt.hist(output[k], label='layer {}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k+1, layer_min, 100-layer_min-layer_max, layer_max), bins=100, color=color2, alpha=alpha)
                plt.vlines(nums_max, 0, 10e4, linestyles='dashed', color=color2, label='threshold 2', alpha=alpha)
            else:
                plt.hist(output[k], label='layer {}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k+1, layer_min, 100-layer_min-layer_max, layer_max), bins=100, color=color2, alpha=alpha)
                plt.vlines(nums_max, 0, 10e4, linestyles='dashed', color=color2, label='threshold 2', alpha=alpha)
                plt.hist(output2[k], label='layer {}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k+1, layer_min2, 100-layer_min2-layer_max2, layer_max2), bins=100, color=color1, alpha=alpha)
                plt.vlines(nums2_max, 0, 10e4, linestyles='dashed', color=color1, label='threshold 1', alpha=alpha)  
            
            # plt.vlines(threshold[str(k)].item() * scale_factor[i], 0, 10e4, linestyles='dashed', color='g', label='updated_threshold')
            # plt.vlines(threshold[str(k)].item(), 0, 1e4, linestyles='dashed', color=color2, label='threshold')
            # plt.vlines(hoyer_line, 0, 10e4, linestyles='dashed',color=color1, label='hoyer_line')
            # plt.vlines(hoyer_sum, 0, 10e5, linestyles='dotted',color='g', label='hoyer_line_sum')

            # mid_num = nums[nums>0.0]
            # mid_num = mid_num[mid_num<nums_max]
            # mid_hoyer = np.sum(mid_num**2) / np.sum(np.abs(mid_num)) if  np.sum(np.abs(mid_num)) > 0 else 0.0
            # mid_hoyers.append(mid_hoyer/nums_max)
            # mid_num = torch.tensor(mid_num)
            # mid_total = mid_num.shape[0]
            # min_scale = mid_num.kthvalue(int(mid_total*0.2)).values.item()
            # max_scale = mid_num.kthvalue(int(mid_total*0.8)).values.item()

            # plt.vlines(min_scale, 0, 10e6, linestyles='dashed', label='min_scale')
            # plt.vlines(max_scale, 0, 10e6, linestyles='dotted', label='max_scale')
            # plt.vlines(mid_hoyer, 0, 10e7, linestyles='dotted', label='mid_hoyer')
        # else:
        #     min /= (len(output.keys())-1)
        #     max /= (len(output.keys())-1)
        #     plt.hist(output2[k], label='{}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k, min, 100-min-max, max), bins=100)
        #     plt.hist(output[k], label='{}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k, min, 100-min-max, max), bins=100)
        plt.legend()
        plt.yscale('log')
    plt.savefig(output_file[:-4] + '_com1.jpg')
    # torch.save(mid_hoyers, 'network_output/my_x_scale_factor_1753')

def plot(mode = 'ann'):
    threshold = {}
    update_thr = {}
    if mode == 'ann':
        pass
        # model = torch.load(os.path.join('trained_models_ann', output_file.split('/')[-1][:-5])+'.pth', map_location='cpu')
        # with open('new_factor_scale/new_factor_x_vgg16_cifar10_1','rb') as f:
        #     scale_factor = pickle.load(f)
    else:
        model = torch.load(os.path.join('trained_snn_models', output_file.split('/')[-1][:-5])+'.pth', map_location='cpu')
        with open('new_factor_scale/new_factor_x_vgg16_cifar10_1','rb') as f:
            scale_factor = pickle.load(f)
        with open('output/ann_max_vgg16_cifar10_0.9', 'rb') as f:
            max_thr_scale = torch.load(f, map_location='cpu')

    # for key in model['state_dict'].keys():
    #     if key[:9] == 'threshold':
    #         threshold[key[11:]] = model['state_dict'][key].cpu()
    if mode == 'snn':
        for i,key in enumerate(sorted(threshold.keys(), key=lambda k: int(k))):
            update_thr[key] = threshold[key]*scale_factor[i]
            print('key: {}, before: {}, update: {}'.format(key, threshold[key], update_thr[key]))
    
    output = torch.load(output_file)
    plt.figure(figsize=(32,32))
    min = max = 0.0
    mid_hoyers = []
    for i,k in enumerate(output.keys()):
        print('k: {}, shape: {}'.format(k, output[k].shape))
        plt.subplot(5,5,i+1)
        nums = np.asarray(output[k])
        nums_max = np.max(nums)
        
        if k != 'total':
            total = nums.shape[0]
            layer_min = (nums<=0).sum() / total * 100.0
            layer_max = (nums>=nums_max).sum()  / total * 100.0
            min += layer_min
            max += layer_max
            if mode == 'snn':
                plt.vlines(update_thr[str(k)].item(), 0, 10e4, linestyles='dashed', color='g', label='updated_thr')
                plt.vlines(max_thr_scale[i], 0, 10e4, linestyles='dashed', color=color2, label='max_thr_scale')
            plt.hist(output[k], label='{}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k, layer_min, 100-layer_min-layer_max, layer_max), bins=100)
            plt.vlines(nums_max, 0, 10e4, linestyles='dashed', color=color2, label='threshold')

            mid_num = nums[nums>0.0]
            mid_num = mid_num[mid_num<nums_max]
            mid_hoyer = np.sum(mid_num**2) / np.sum(np.abs(mid_num)) if  np.sum(np.abs(mid_num)) > 0 else 0.0
            mid_hoyers.append(mid_hoyer/nums_max)
        else:
            min /= (len(output.keys())-1)
            max /= (len(output.keys())-1)
            plt.hist(output[k], label='{}: 0: {:.2f}%, (0,thr): {:.2f}%, thr: {:.2f}%'.format(k, min, 100-min-max, max), bins=100)
        plt.legend()
        plt.yscale('log')
    plt.savefig(output_file[:-4] + '_com.jpg')
    # torch.save(mid_hoyers, 'network_output/my_x_scale_factor_1753')


def plot_heatmap():  
    output = torch.load(output_file)
    plt.figure(figsize=(32,32))
    min = max = 0.0
    mid_hoyers = []
    # 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
    # bins_num = [2**(32*32), 2**(32*32), 2**(16*16), 2**(16*16), 2**(8*8), 2**(8*8), 2**(8*8), 2**(4*4), 2**(4*4), 2**(4*4), 2**(2*2), 2**(2*2), 2**(2*2)]
    bins_num = np.array([(32*32), (32*32), (16*16), (16*16), (8*8), (8*8), (8*8), (4*4), (4*4),(4*4), (2*2), (2*2), (2*2)])
    range_num = 2**(bins_num)
    print(range_num)
    for i,k in enumerate(output.keys()):
        plt.subplot(4,4,i+1)
        nums = np.asarray((output[k]))
        print('shape: {}'.format(nums.shape))
        # print(nums[:,0])
        if k != 'total':
            # plt.hist(output[k], label='{}'.format(k), bins=1024)
            if int(k) <= 42:
                sns.heatmap(nums,  cmap="YlGnBu")
            # elif int(k) <= 42:

            #     plt.hist(nums, label='{}'.format(k), bins=128)
            #     # plt.yscale('log') range=(1,range_num[i])
            #     plt.legend()
            else:
                plt.hist(nums, label='{}'.format(k), bins=4096)
                plt.yscale('log')
                plt.legend()
            
    plt.savefig(output_file[:-4] + '_heatmap.jpg')
    # torch.save(mid_hoyers, 'network_output/my_x_scale_factor_1753')

if __name__ == '__main__':
    # plot_heatmap()
    # subplot()
    # plot()
    combine_plot()
    # plot('snn')
