import os
import time
import argparse

import numpy as np

import torch
import torch.nn as nn

from models.channel_selection import ChannelSelection
from helper import get_model, create_dataloader, count_parameters
from checkpointer import Checkpointer
from main import run_test


def get_bn_wight_thresh(model, pruned_percent):
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
    bn_weight_thresh = y[int(bn_channel_count * pruned_percent)]

    return bn_weight_thresh


def get_file_mb(file_path):
    return os.stat(file_path).st_size / 1024 / 1024


def main(args):
    model = get_model(args.arch)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    _, testloader = create_dataloader(128)

    ckpt = Checkpointer(model, save_dir=args.ckpt_dir)
    last_ckpt_path = ckpt.load()
    origin_ckpt_size = get_file_mb(last_ckpt_path)
    model = model.to(device)

    # gpu warm up for accuracy time measure
    run_test(model, testloader, nn.CrossEntropyLoss(), device)
    torch.cuda.synchronize()

    # 测试原始模型准确率
    start_time = time.time()
    origin_acc, origin_loss = run_test(model, testloader, nn.CrossEntropyLoss(), device)
    torch.cuda.synchronize()
    origin_process_time = time.time() - start_time
    origin_params_count = count_parameters(model)

    bn_weight_thresh = get_bn_wight_thresh(model, args.percent).to(device)

    # 每层 BN 层保留的通道数
    cfg = []
    # 每层 BN 层通道的 mask，记录哪些通道要剪枝
    bn_channel_mask = []
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(bn_weight_thresh).float().to(device)

            remain_channel_count = int(torch.sum(mask))
            channel_count = mask.shape[0]
            pruned_channel_count = channel_count - remain_channel_count

            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)

            cfg.append(remain_channel_count)
            bn_channel_mask.append(mask.clone())
            print(f"BN layer index: {idx}\t"
                  f"pruned {(pruned_channel_count / channel_count) * 100:.2f}%"
                  f"({pruned_channel_count}/{channel_count})")

    print(f"cfg {len(cfg)}: {cfg}")

    # simple test model after Pre-processing prune (simple set BN scales to zeros)
    # acc, loss = run_test(model, testloader, nn.CrossEntropyLoss(), device)
    # print(f"acc:{acc} loss: {loss}")

    pruned_model = get_model(args.arch, cfg)
    pruned_model.to(device)

    old_modules = list(model.modules())
    pruned_modules = list(pruned_model.modules())

    layer_idx_in_cfg = 0
    pre_bn_channel_mask = torch.ones(3)
    next_bn_channel_mask = bn_channel_mask[layer_idx_in_cfg]
    conv_count = 0

    for layer_idx in range(len(old_modules)):
        # 被剪枝的模型和原来的模型，layer 的数量是一样的，只是每层的 channel 可能不同
        m0 = old_modules[layer_idx]
        m1 = pruned_modules[layer_idx]

        if isinstance(m0, nn.BatchNorm2d):
            next_bn_remain_channel_idxes = np.squeeze(np.argwhere(next_bn_channel_mask.cpu().numpy()))
            if next_bn_remain_channel_idxes.size == 1:
                next_bn_remain_channel_idxes = np.resize(next_bn_remain_channel_idxes, (1,))

            if isinstance(old_modules[layer_idx + 1], ChannelSelection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = pruned_modules[layer_idx + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[next_bn_remain_channel_idxes.tolist()] = 1.0

                layer_idx_in_cfg += 1
                pre_bn_channel_mask = next_bn_channel_mask.clone()
                if layer_idx_in_cfg < len(bn_channel_mask):  # do not change in Final FC
                    next_bn_channel_mask = bn_channel_mask[layer_idx_in_cfg]
            else:
                # BN 层只获得 mask 为 1 的 channel
                m1.weight.data = m0.weight.data[next_bn_remain_channel_idxes.tolist()].clone()
                m1.bias.data = m0.bias.data[next_bn_remain_channel_idxes.tolist()].clone()
                m1.running_mean = m0.running_mean[next_bn_remain_channel_idxes.tolist()].clone()
                m1.running_var = m0.running_var[next_bn_remain_channel_idxes.tolist()].clone()

                layer_idx_in_cfg += 1
                pre_bn_channel_mask = next_bn_channel_mask.clone()
                if layer_idx_in_cfg < len(bn_channel_mask):  # do not change in Final FC
                    next_bn_channel_mask = bn_channel_mask[layer_idx_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                # 对应 conv1
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue

            if isinstance(old_modules[layer_idx - 1], ChannelSelection) \
                    or isinstance(old_modules[layer_idx - 1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                # 当前卷积层上方第一个 BN 保留的通道索引
                pre_bn_remain_channel_idxes = np.squeeze(np.argwhere(pre_bn_channel_mask.cpu().numpy()))
                # 当前卷积层下方第一个 BN 保留的通道索引
                next_bn_remain_channel_idxes = np.squeeze(np.argwhere(next_bn_channel_mask.cpu().numpy()))
                # print('In shape: {:d}, Out shape {:d}.'.format(pre_bn_remain_channel_idxes.size,
                #                                                next_bn_remain_channel_idxes.size))

                if pre_bn_remain_channel_idxes.size == 1:
                    pre_bn_remain_channel_idxes = np.resize(pre_bn_remain_channel_idxes, (1,))
                if next_bn_remain_channel_idxes.size == 1:
                    next_bn_remain_channel_idxes = np.resize(next_bn_remain_channel_idxes, (1,))

                w1 = m0.weight.data[:, pre_bn_remain_channel_idxes.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[next_bn_remain_channel_idxes.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            pre_bn_remain_channel_idxes = np.squeeze(np.argwhere(pre_bn_channel_mask.cpu().numpy()))
            if pre_bn_remain_channel_idxes.size == 1:
                pre_bn_remain_channel_idxes = np.resize(pre_bn_remain_channel_idxes, (1,))

            # 因为 avg pool 已经把 feature map 转成了 [B, C, 1, 1] 所以这里可以直接通 channel 的 mask
            m1.weight.data = m0.weight.data[:, pre_bn_remain_channel_idxes].clone()
            m1.bias.data = m0.bias.data.clone()

    # cfg 用来构造 model
    # state_dict 用来恢复参数
    pruned_model_path = os.path.join(args.save_dir, 'pruned.pth.tar')
    torch.save({
        'cfg': cfg,
        'state_dict': pruned_model.state_dict()
    }, pruned_model_path)

    pruned_ckpt_size = get_file_mb(pruned_model_path)

    start_time = time.time()
    acc, loss = run_test(pruned_model, testloader, nn.CrossEntropyLoss(), device)
    torch.cuda.synchronize()
    pruned_model_time = time.time() - start_time
    pruned_params_count = count_parameters(pruned_model)

    print(f"Pruned rate: {args.percent}")
    print(f"Model acc:{origin_acc:.3f} loss: {origin_loss:.3f} time: {origin_process_time:.3f}s")
    print(f"Pruned model acc:{acc:.3f} loss: {loss:.3f} time: {pruned_model_time:.3f}s")

    print(f"Time reduce: {(origin_process_time - pruned_model_time) / origin_process_time * 100:.4f}% "
          f"({origin_process_time:.2f}s -> {pruned_model_time:.2f}s)\n"
          f"Model params reduce: {(origin_params_count - pruned_params_count) / origin_params_count * 100:.4f}% "
          f"({origin_params_count:.2f} -> {pruned_params_count:.2f})\n"
          f"Model size reduce: {(origin_ckpt_size - pruned_ckpt_size) / origin_ckpt_size * 100:.4f}% "
          f"({origin_ckpt_size:.2f}mb->{pruned_ckpt_size:.2f}mb)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--percent', type=float, default=0.4,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ckpt_dir', default='./ckpts/preact_res20/best_acc', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save_dir', default='./ckpts/pruned', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    parser.add_argument('--arch', default='preact_res20', type=str,
                        choices=['preact_res20'],
                        help='architecture to use')

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        print(f"{args.ckpt_dir} not exists")
        exit(-1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
