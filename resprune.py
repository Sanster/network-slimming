import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from helper import get_model, create_dataloader
from checkpointer import Checkpointer
from main import run_test


def main(args):
    model = get_model(args.arch)
    _, testloader = create_dataloader(8)

    ckpt = Checkpointer(model, save_dir=args.ckpt_dir)
    ckpt.load()

    # simple test model after Pre-processing prune (simple set BN scales to zeros)
    acc, loss = run_test(model, testloader, nn.CrossEntropyLoss(), 'cpu')
    print(f"acc:{acc} loss: {loss}")

    bn_channel_count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_channel_count += m.weight.data.shape[0]

    bn = torch.zeros(bn_channel_count)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(bn_channel_count * args.percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned / bn_channel_count
    print(f'Pre-processing Successful: {pruned_ratio}')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)
    acc, loss = run_test(model, testloader, nn.CrossEntropyLoss(), 'cpu')
    print(f"acc:{acc} loss: {loss}")


# newmodel = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)
# if args.cuda:
#     newmodel.cuda()
#
# num_parameters = sum([param.nelement() for param in newmodel.parameters()])
# savepath = os.path.join(args.save, "prune.txt")
# with open(savepath, "w") as fp:
#     fp.write("Configuration: \n" + str(cfg) + "\n")
#     fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
#     fp.write("Test accuracy: \n" + str(acc))
#
# old_modules = list(model.modules())
# new_modules = list(newmodel.modules())
# layer_id_in_cfg = 0
# start_mask = torch.ones(3)
# end_mask = cfg_mask[layer_id_in_cfg]
# conv_count = 0
#
# for layer_id in range(len(old_modules)):
#     m0 = old_modules[layer_id]
#     m1 = new_modules[layer_id]
#     if isinstance(m0, nn.BatchNorm2d):
#         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#         if idx1.size == 1:
#             idx1 = np.resize(idx1, (1,))
#
#         if isinstance(old_modules[layer_id + 1], channel_selection):
#             # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
#             m1.weight.data = m0.weight.data.clone()
#             m1.bias.data = m0.bias.data.clone()
#             m1.running_mean = m0.running_mean.clone()
#             m1.running_var = m0.running_var.clone()
#
#             # We need to set the channel selection layer.
#             m2 = new_modules[layer_id + 1]
#             m2.indexes.data.zero_()
#             m2.indexes.data[idx1.tolist()] = 1.0
#
#             layer_id_in_cfg += 1
#             start_mask = end_mask.clone()
#             if layer_id_in_cfg < len(cfg_mask):
#                 end_mask = cfg_mask[layer_id_in_cfg]
#         else:
#             m1.weight.data = m0.weight.data[idx1.tolist()].clone()
#             m1.bias.data = m0.bias.data[idx1.tolist()].clone()
#             m1.running_mean = m0.running_mean[idx1.tolist()].clone()
#             m1.running_var = m0.running_var[idx1.tolist()].clone()
#             layer_id_in_cfg += 1
#             start_mask = end_mask.clone()
#             if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
#                 end_mask = cfg_mask[layer_id_in_cfg]
#     elif isinstance(m0, nn.Conv2d):
#         if conv_count == 0:
#             m1.weight.data = m0.weight.data.clone()
#             conv_count += 1
#             continue
#         if isinstance(old_modules[layer_id - 1], channel_selection) or isinstance(old_modules[layer_id - 1],
#                                                                                   nn.BatchNorm2d):
#             # This convers the convolutions in the residual block.
#             # The convolutions are either after the channel selection layer or after the batch normalization layer.
#             conv_count += 1
#             idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#             idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#             print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#             if idx0.size == 1:
#                 idx0 = np.resize(idx0, (1,))
#             if idx1.size == 1:
#                 idx1 = np.resize(idx1, (1,))
#             w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#
#             # If the current convolution is not the last convolution in the residual block, then we can change the
#             # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
#             if conv_count % 3 != 1:
#                 w1 = w1[idx1.tolist(), :, :, :].clone()
#             m1.weight.data = w1.clone()
#             continue
#
#         # We need to consider the case where there are downsampling convolutions.
#         # For these convolutions, we just copy the weights.
#         m1.weight.data = m0.weight.data.clone()
#     elif isinstance(m0, nn.Linear):
#         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#         if idx0.size == 1:
#             idx0 = np.resize(idx0, (1,))
#
#         m1.weight.data = m0.weight.data[:, idx0].clone()
#         m1.bias.data = m0.bias.data.clone()
#
# torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
#
# print(newmodel)
# model = newmodel
# test(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ckpt_dir', default='./ckpts/preact_res18_sr/best_acc', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save_dir', default='./ckpts/pruned', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument('--arch', default='preact_res18', type=str,
                        choices=['res50', 'preact_res18', 'preact_res34', 'preact_res50'],
                        help='architecture to use')

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        print(f"{args.ckpt_dir} not exists")
        exit(-1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
