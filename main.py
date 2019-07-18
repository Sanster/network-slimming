import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import resnet50
from checkpointer import Checkpointer, BestCheckpointer


def parse_args():
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
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
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
    
    return args


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

global_train_step = 0

def create_dataloader(batch_size):
    norm = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    to_tensor = torchvision.transforms.ToTensor()

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        to_tensor,
        norm
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=10)

    test_transform = torchvision.transforms.Compose([
        to_tensor,
        norm
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=10)
    return trainloader, testloader

def create_summary_writer(save_dir: Path):
    train_writer = SummaryWriter(str(save_dir / 'train'))
    test_writer = SummaryWriter(str(save_dir / 'test'))
    return train_writer, test_writer


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    trainloader, testloader = create_dataloader(batch_size=args.batch_size)
    train_writer, test_writer = create_summary_writer(args.log_dir)

    model = resnet50(pretrained=False, num_classes=10)
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    saver = Checkpointer(model, optimizer, scheduler, str(args.ckpt_dir))

    best_acc_saver = BestCheckpointer(model, optimizer, scheduler, str(args.ckpt_dir / 'best_acc'), 3, 0)
    lowest_loss_saver = BestCheckpointer(model, optimizer, scheduler, str(args.ckpt_dir / 'lowest_loss'), 3, float('-inf'))

    total_epoch = args.epochs
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        train_epoch(model, trainloader, criterion, optimizer, train_writer, device, epoch)
        scheduler.step()
        acc, loss = run_test(model, testloader, criterion, test_writer, device, epoch)

        ckpt_name = f'ckpt_{global_train_step}_acc{acc:.3f}_loss{loss:.4f}.pth'
        saver.save(ckpt_name)

        if acc >= best_acc_saver.best_value:
            best_acc_saver.best_value = acc
            best_acc_saver.save(ckpt_name)
        
        if lowest_loss_saver.best_value <= loss:
            lowest_loss_saver.best_value
            lowest_loss_saver.save(ckpt_name)

    print('Finished Training')


def train_epoch(model, dataloader, criterion, optimizer, writer, device, epoch):
    global global_train_step
    loop = tqdm(dataloader)
    for i, (inputs, labels) in enumerate(loop):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

        global_train_step += 1
        if global_train_step % 5 == 0:
            writer.add_scalar('loss', loss.item(), global_train_step)


def run_test(model, dataloader, criterion, writer, device, epoch):
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

    writer.add_scalar('loss', running_loss, global_train_step)
    acc = correct / total * 100
    writer.add_scalar('acc', acc, global_train_step)
    print(f"Epoch {epoch} test acc: {acc:.3f}({correct}/{total})  loss: {running_loss}")

    return acc, running_loss


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

# # additional subgradient descent on the sparsity-induced penalty term
# def updateBN():
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1
#
#
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         pred = output.data.max(1, keepdim=True)[1]
#         loss.backward()
#         if args.sr:
#             updateBN()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.data[0]))
#
#
# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.cross_entropy(output, target, size_average=False).data[0]  # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#     return correct / float(len(test_loader.dataset))
#
#
# def save_checkpoint(state, is_best, filepath):
#     torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
#     if is_best:
#         shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
#
#
# best_prec1 = 0.
# for epoch in range(args.start_epoch, args.epochs):
#     if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= 0.1
#     train(epoch)
#     prec1 = test()
#     is_best = prec1 > best_prec1
#     best_prec1 = max(prec1, best_prec1)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'best_prec1': best_prec1,
#         'optimizer': optimizer.state_dict(),
#     }, is_best, filepath=args.save)
#
# print("Best accuracy: " + str(best_prec1))
