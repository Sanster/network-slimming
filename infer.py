import os
import time
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from helper import get_model, create_dataloader
from checkpointer import Checkpointer
from main import run_test


def main(args):
    checkpoint = torch.load(args.ckpt_path)
    cfg = checkpoint.get('cfg')
    if cfg:
        print(f'Load cfg from checkpoint: {cfg}')

    model = get_model(args.arch, cfg)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    _, testloader = create_dataloader(args.batch_size)

    ckpt = Checkpointer(model)
    ckpt.load(args.ckpt_path)
    model = model.to(device)

    # gpu warm up for accuracy time measure
    run_test(model, testloader, nn.CrossEntropyLoss(), device)
    torch.cuda.synchronize()

    # 测试原始模型准确率
    process_time = 0
    acc = 0
    loss = 0
    count = 10
    for i in tqdm(range(count)):
        start_time = time.time()
        tmp_acc, tmp_loss = run_test(model, testloader, nn.CrossEntropyLoss(), device)
        torch.cuda.synchronize()
        process_time += (time.time() - start_time)
        acc += tmp_acc
        loss += tmp_loss

    print(f"Average acc: {acc / count}")
    print(f"Average loss: {loss / count}")
    print(f"Average process time: {process_time / count}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ckpt_path', default='./ckpts/preact_res20_sr/best_acc', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--arch', default='preact_res20', type=str,
                        choices=['preact_res20'],
                        help='architecture to use')

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f"{args.ckpt_path} not exists")
        exit(-1)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
