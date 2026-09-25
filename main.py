# coding=utf-8
from __future__ import print_function

import math
import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import utils.Util_visual as visual

from trainer.fixbi_trainer import train_fixbi
from src.dataset import get_dataset
import src.utils_1 as utils

parser = argparse.ArgumentParser(description="FIXBI EXPERIMENTS")
parser.add_argument('-db_path', help='gpu number', type=str, default='/database')
parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test')
parser.add_argument('-source', help='source', type=str, default='amazon')  # [amazon  webcam  dslr] [train, validation]
parser.add_argument('-target', help='target', type=str, default='webcam')  # ['Art', 'Clipart', 'Product', 'Real_World']
parser.add_argument('-workers', default=0, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0,1')
parser.add_argument('-epochs', default=200, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-th', default=5.5, type=float, help='Threshold')
parser.add_argument('-bim_start', default=150, type=int, help='Bidirectional Matching')
parser.add_argument('-fix_start', default=40, type=int, help='FixMatch-start')
parser.add_argument('-sp_start', default=0, type=int, help='Self-Penalization')
parser.add_argument('-fix_sp', default=10, type=int, help='fix_sp_loss')
parser.add_argument('-style_match', default=0, type=int, help='Style_Match_loss')
parser.add_argument('-mix_start', default=20, type=int, help='Select MixUp')
parser.add_argument('-con_start', default=100, type=int, help='Consistency Regularization')
parser.add_argument('-lam_sd', default=0.7, type=float, help='Source Dominant Mixup ratio')
parser.add_argument('-lam_td', default=0.3, type=float, help='Target Dominant Mixup ratio')
parser.add_argument('-padain', type=float, default=0)
parser.add_argument('-threshold', type=float, default=4.5)


# parser.add_argument('-is_Padain', type=bool, default=True)
# parser.add_argument('no_Padain', type=bool, default=False)
# 正常运行参数
# -gpu=1
# -db_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/datasets
# -baseline_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/DeepDA/baseline_path
# -save_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/profix/save_path
# -padain=0


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))
    print(args)
    utils.set_random_seed(seed=2019)

    num_classes, resnet_type = utils.get_data_info()
    src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
    tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)

    src_train_loader = torchdata.DataLoader(src_trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_train_loader = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True, drop_last=False)

    lr, l2_decay, momentum, nesterov = utils.get_train_info()
    net_sd, head_sd, classifier_sd = utils.get_net_info(num_classes)  # 该容器通过在批处理维度中分组，将输入分割到指定的设备上

    learnable_params_sd = list(net_sd.parameters()) + list(head_sd.parameters()) + list(
        classifier_sd.parameters())  # 将分组的参数合并

    optimizer_sd = optim.SGD(learnable_params_sd, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)

    sp_param_sd = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)  # 将tensor变成可训练的

    optimizer_sd.add_param_group({"params": [sp_param_sd], "lr": lr})

    ce = nn.CrossEntropyLoss().cuda()
    mse = nn.MSELoss().cuda()

    net_sd, head_sd, classifier_sd = utils.load_net(args, net_sd, head_sd, classifier_sd)

    loaders = [src_train_loader, tgt_train_loader]

    optimizers = [optimizer_sd]
    models_sd = [net_sd, head_sd, classifier_sd]

    sp_params = sp_param_sd
    losses = [ce, mse]

    lam = 0
    pre_lam = 0
    par_lam = 0

    for epoch in range(args.epochs):

        if (epoch - 1) % 10 == 0:
            lam_ratio = 1 / math.exp((epoch - 1) / 100)
            lam = 1 - lam_ratio
            if lam == 0:
                lam = 0.1
            if par_lam != lam:
                pre_lam = par_lam
                par_lam = lam

        # train_fixbi(args, loaders, optimizers, models_sd, sp_params, losses, epoch, lam, pre_lam)

        # utils.evaluate(nn.Sequential(*models_sd), tgt_test_loader)

        # utils.final_eval(nn.Sequential(*models_sd), tgt_test_loader)
        # visual.final_eval2(nn.Sequential(*models_sd), tgt_test_loader)

        utils.getTsne(nn.Sequential(*models_sd), tgt_test_loader)

        # utils.test(nn.Sequential(*models_sd), tgt_test_loader)

        # utils.save_net(args, models_sd, 'sdm')



if __name__ == "__main__":
    main()
