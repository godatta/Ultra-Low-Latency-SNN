import torch
import copy


# PATH = 'backup/epoch_107_9011_drop9.pth'
# # PATH = 'trained_snn_models/snn_vgg16_cifar10_5_202203261804/pruned_epoch_3.pth'
# model = torch.load(PATH)
# thresholds = model['thresholds']

# features_layer_output = torch.load('output/features_layer_output')
# classifier_layer_output = torch.load('output/classifier_layer_output')


# print(layer_output.keys())
# print(layer_output[0].keys())

# res = copy.deepcopy(classifier_layer_output[0])
def get_total_uotput(features_layer_output, classifier_layer_output):
    total_output = {}

    for batchid in features_layer_output.keys():
        if batchid == 8:
            break
        for layer in features_layer_output[batchid].keys():
            if layer not in total_output:
                total_output[layer] = features_layer_output[batchid][layer].view(-1)
                # print(f'In features: layer {layer} shape: {total_output[layer].shape[0]}')
            else:
                total_output[layer] = torch.hstack((total_output[layer], features_layer_output[batchid][layer].view(-1)))
        for layer in classifier_layer_output[batchid].keys():
            if layer == 19:
                break
            if layer not in total_output:
                total_output[layer] = classifier_layer_output[batchid][layer].view(-1)
                # print(f'In classifier: layer {layer} shape: {total_output[layer].shape[0]}')
            else:
                total_output[layer] = torch.hstack((total_output[layer], classifier_layer_output[batchid][layer].view(-1)))
    return total_output

def train(features_layer_output, classifier_layer_output, min_thr=0.1, max_thr=0.9):
    total_output = get_total_uotput(features_layer_output, classifier_layer_output)

    min_scale = []
    max_scale = []

    relu_num = 0
    total_num = 0

    for layer in total_output.keys():
        len_ori = total_output[layer].shape[0]
        pos_output = total_output[layer][total_output[layer]>0.0]
        pos_output = pos_output[pos_output<1.0]
        len_pos = pos_output.shape[0]
        min = pos_output.kthvalue(int(len_pos*min_thr)).values.item()
        min_scale.append(min)
        max = pos_output.kthvalue(int(len_pos*max_thr)).values.item()
        max_scale.append(max)
        relu_num_layer = (len_ori - (total_output[layer] < min).sum() - (total_output[layer] > max).sum())
        relu_num += relu_num_layer
        total_num += len_ori
        print('In layer {}, # of elements is {}, # of pos elements is {:.4f}%, # of relu is {:.4f}%, min is {:.6f}, max is {:.6f}'.format(
            layer,
            len_ori,
            relu_num_layer/len_pos*100.0,
            relu_num_layer/len_ori*100.0,
            min,
            max,
        ))
    print('total relu rate: {:.4f}%'.format(100.0*relu_num/total_num))
    torch.save(min_scale, f'output/ann_min_scale_vgg16_cifar10_{str(min_thr)}')
    torch.save(max_scale, f'output/ann_max_scale_vgg16_cifar10_{str(max_thr)}')

def test(features_layer_output, classifier_layer_output, min_scale, max_scale):
    total_output = get_total_uotput(features_layer_output, classifier_layer_output)

    relu_num = 0
    total_num = 0

    for i,layer in enumerate(total_output.keys()):
        len_ori = total_output[layer].shape[0] 
        num0 = (total_output[layer] < 0.0).sum() # x < 0
        num1 = (total_output[layer] < min_scale[i]).sum() # x < thr*min_scale
        num2 = (total_output[layer] > max_scale[i]).sum() # x > thr*max_scale
        num3 = (total_output[layer] > 1.0).sum() # x > thr

        len_pos = len_ori - num0 - num3
        len_relu = len_ori-num1-num2
        relu_num += len_relu
        total_num += len_ori

        print('In layer {}, # of elements is {}, relu rate in pos elements is {:.4f}%, relu rate is {:.4f}%, min is {}, max is {}'.format(
            layer,
            len_ori,
            len_relu/len_pos*100.0,
            len_relu/len_ori*100.0,
            min_scale[i],
            max_scale[i]
        ))
    print('total relu rate: {:.4f}%'.format(float(relu_num/total_num*100.0)))

def tdbn_train(total_output, min_thr=0.1, max_thr=0.9):

    min_scale = []
    max_scale = []

    for layer in total_output.keys():
        len_ori = total_output[layer].shape[0]
       
        min = total_output[layer].kthvalue(int(len_ori*min_thr)).values.item()
        min_scale.append(min)
        max = total_output[layer].kthvalue(int(len_ori*max_thr)).values.item()
        max_scale.append(max)
       
        print('In layer {}, # of elements is {}, min is {:.6f}, max is {:.6f}'.format(
            layer,
            len_ori,
            min,
            max,
        ))
    torch.save(min_scale, f'output/ann_min_scale_vgg16_cifar10_tdbn_{str(min_thr)}')
    torch.save(max_scale, f'output/ann_max_scale_vgg16_cifar10_tdbn_{str(max_thr)}')

if __name__ == '__main__':
    # feat_filename = 'output/features_layer_output'
    # cls_filename = 'output/classifier_layer_output'

    # feat_filename = 'output/ann_features_layer_output'
    # cls_filename = 'output/ann_classifier_layer_output'

    # features_layer_output = torch.load(feat_filename)
    # classifier_layer_output = torch.load(cls_filename)
    # min_scale = torch.load('output/ann_min_scale_vgg16_cifar10_0.1')
    # max_scale = torch.load('output/ann_max_scale_vgg16_cifar10_0.9')
    # train(features_layer_output, classifier_layer_output, 0.3, 0.7)
    # test(features_layer_output, classifier_layer_output, min_scale, max_scale)

    filename = 'output/ann_tdbn_layer_output'
    total_output = torch.load(filename)
    tdbn_train(total_output, 0.4, 0.6)