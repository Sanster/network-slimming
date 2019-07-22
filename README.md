# Network Slimming (Pytorch 1.0)

This repository contains an pytorch implementation for the following paper 
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  


- python 3.5+
- pytorch 1.1.0

## Channel Selection Layer
`ChannelSelection` layer is used to help the pruning of ResNet. This layer stores a parameter indexes which is initialized to an all-1 vector.
During pruning, it will set some places to 0 which correspond to the pruned channels.

## Baseline
Train preactivation resnet20 as baseline. You can see distribution of `BatchNorm2d` weight in tensorboard.
```bash
python3 main.py --arch=preact_res20 --tag=baseline
```

## Train with Sparsity
Add L1 regularization to BatchNorm2d layer's weight

```bash
python3 main.py -sr --s 0.0001 --arch=preact_res20
```

`--s` will be used in `update_bn` function
```python
def update_bn(model, s=0.0001):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1
```

Best accuracy model in `./ckpts/preact_res20/best_acc`

You can compare distribution of `BatchNorm2d` weight in tensorboard.

## Prune model trained with sparsity
Run following command and pruned model will be saved as `./ckpts/pruned/prect/preact_res20_pruned.pth.tar`.

```bash
python3 resprune.py --ckpt_dir=./ckpts/preact_res20/best_acc --arch=preact_res20

```


## Fine tune pruned model
```bash
python3 main.py --refine=./ckpts/pruned/prect/preact_res20_pruned.pth.tar --arch=preact_res20 --tag=fine_tune_pruned
```

