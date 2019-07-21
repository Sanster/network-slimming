import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm

from checkpointer import Checkpointer, BestCheckpointer
from helper import get_model, create_dataloader

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

global_train_step = 0


def create_summary_writer(save_dir: Path):
    train_writer = SummaryWriter(str(save_dir / 'train'))
    test_writer = SummaryWriter(str(save_dir / 'test'))
    return train_writer, test_writer


def update_bn(model, s=0.0001):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    trainloader, testloader = create_dataloader(batch_size=args.batch_size)
    train_writer, test_writer = create_summary_writer(args.log_dir)

    model = get_model(args.arch)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    saver = Checkpointer(model, optimizer, scheduler, str(args.ckpt_dir))

    best_acc_saver = BestCheckpointer(model, optimizer, scheduler, str(args.ckpt_dir / 'best_acc'), 3, 0)
    lowest_loss_saver = BestCheckpointer(model, optimizer, scheduler, str(args.ckpt_dir / 'lowest_loss'), 3,
                                         float('-inf'))

    total_epoch = args.epochs
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        train_epoch(model, trainloader, criterion, optimizer, train_writer, device, epoch)
        scheduler.step()
        acc, test_loss = run_test(model, testloader, criterion, device)

        test_writer.add_scalar('loss', test_loss, global_train_step)
        test_writer.add_scalar('acc', acc, global_train_step)
        print(f"Epoch {epoch} test acc: {acc:.3f} loss: {test_loss}")

        ckpt_name = f'ckpt_{global_train_step}_acc{acc:.3f}_loss{test_loss:.4f}.pth'
        saver.save(ckpt_name)

        if acc >= best_acc_saver.best_value:
            best_acc_saver.best_value = acc
            best_acc_saver.save(ckpt_name)

        if lowest_loss_saver.best_value <= test_loss:
            lowest_loss_saver.best_value = test_loss
            lowest_loss_saver.save(ckpt_name)

    print('Finished Training')


def train_epoch(model, dataloader, criterion, optimizer, writer, device, epoch):
    model.train()
    global global_train_step
    loop = tqdm(dataloader)
    total = 0
    correct = 0
    for i, (inputs, labels) in enumerate(loop):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        update_bn(model)
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())

        global_train_step += 1

    acc = correct / total * 100
    writer.add_scalar('loss', loss.item(), global_train_step)
    writer.add_scalar('acc', acc, global_train_step)


def run_test(model, dataloader, criterion, device):
    model.eval()
    print("run test...")
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

    running_loss /= len(dataloader)
    acc = correct / total * 100

    return acc, running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--ckpt-dir', default='./ckpts', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--log-dir', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='res50', type=str,
                        choices=['res50', 'preact_res18', 'preact_res34', 'preact_res50'],
                        help='architecture to use')
    parser.add_argument('--tag', default='')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    args.log_dir = Path(args.log_dir)
    args.ckpt_dir = Path(args.ckpt_dir)

    sub_dir = f"{args.arch}{'_sr' if args.sr else ''}"
    sub_dir += f"_{args.tag}" if args.tag else ''

    args.log_dir /= sub_dir
    args.ckpt_dir /= sub_dir

    torch.backends.cudnn.benchmark = True
    main(args)
