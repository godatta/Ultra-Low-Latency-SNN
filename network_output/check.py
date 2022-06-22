import torch
import numpy as np
import matplotlib.pyplot as plt

# output_file = 'network_output/snn_vgg16_cifar10_1_202205221408_test'
# output = torch.load(output_file)
# print(len(output['total']))
# output = np.asarray(output['total'])
# plt.figure(figsize=(32,32))
# plt.hist(output, bins=100)
# plt.yscale('log')
# plt.savefig(output_file[:-4] + '.jpg')


output_file = 'network_output/ann_vgg16_cifar10_4.0_0.2lr_decay_test'
output = torch.load(output_file)
for k in output.keys():
    out = torch.tensor(output[k])
    num_mean = torch.mean(out[out>0.0])
    hoyer = torch.mean(torch.sum(out**2)) / torch.mean(torch.sum(torch.abs(out)))
    hoyer_mean = torch.mean(out[out>=hoyer])
    new_mean = torch.sum(out) / len(out[out>=hoyer])
    print('{} layer mean: {:.4f}, hoyer: {:.4f}, hoyer_mean: {:.4f}, new_mean:{:.4f}'.format(k, num_mean, hoyer, hoyer_mean, new_mean))