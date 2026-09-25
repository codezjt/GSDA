# coding=utf-8
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

# import network.models_Ada as models
# import network.models_Ada_noAdain as models


def get_data_info():
    resnet_type = 50
    num_classes = 31  # Office-31
    # num_classes = 65  # Office-home
    return num_classes, resnet_type


def get_net_info(num_classes, with_permute_adain=False, p_adain=None):

    net = torch.nn.parallel.DataParallel(models.ResNet50(with_permute_adain=with_permute_adain, p_adain=p_adain).encoder).cuda()  #该容器通过在批处理维度中分组，将输入分割到指定的设备上，
    classifier = torch.nn.parallel.DataParallel(nn.Linear(256, num_classes)).cuda()
    head = torch.nn.parallel.DataParallel(models.Head()).cuda()


    return net, head, classifier


def get_train_info():
    lr = 0.001
    l2_decay = 5e-4
    momentum = 0.9
    nesterov = False
    return lr, l2_decay, momentum, nesterov


def load_net(args, net, head, classifier):
    print("Load pre-trained baseline model !")
    save_folder = args.baseline_path
    net.module.load_state_dict(torch.load(save_folder + '/net.pt'), strict=False)
    head.module.load_state_dict(torch.load(save_folder + '/head.pt'), strict=False)
    classifier.module.load_state_dict(torch.load(save_folder + '/classifier.pt'), strict=False)
    return net, head, classifier


def save_net(args, models, type):
    save_folder = args.save_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    net, head, classifier = models[0], models[1], models[2]

    torch.save(net.module.state_dict(), save_folder + '/' + 'net_' + str(type) + '.pt')
    torch.save(head.module.state_dict(), save_folder + '/' + 'head_' + str(type) + '.pt')
    torch.save(classifier.module.state_dict(), save_folder + '/' + 'classifier_' + str(type) + '.pt')


def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def evaluate(models, loader):
    start = time.time()
    total = 0
    correct = 0
    set_model_mode('eval', [models])
    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
            tgt_preds = models(tgt_imgs)
            pred = tgt_preds.argmax(dim=1, keepdim=True)
            correct += pred.eq(tgt_labels.long().view_as(pred)).sum().item()
            total += tgt_labels.size(0)

    print('Accuracy: {:.2f}%'.format((correct / total) * 100))
    print("Eval time: {:.2f}".format(time.time() - start))
    set_model_mode('train', [models])


def get_sp_loss(input, target, temp):
    criterion = nn.NLLLoss(reduction='none').cuda()
    loss = torch.mul(criterion(torch.log(1 - F.softmax(input / temp, dim=1)), target.detach()), 1).mean()
    return loss


def get_target_preds(args, x):
    top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)   # 按行返回每行最大的值和下标
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold = top_mean - args.th * top_std
    return top_label, top_prob, threshold   # threshold置信度阈值


def mixup_criterion_hard(pred, y_a, y_b, lam):   # FL
    criterion = nn.CrossEntropyLoss().cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio):
    alpha = 1
    ratio = np.random.beta(alpha, alpha)
    ratio = 0.8
    mixed_x = ratio * src_imgs + (1 - ratio) * tgt_imgs
    mixed_x = net(mixed_x)
    loss = mixup_criterion_hard(mixed_x, src_labels.detach(), tgt_pseudo.detach(), ratio)
    return loss


def final_eval(models_sd, tgt_test_loader):
    total = 0
    correct = 0
    set_model_mode('eval', [*models_sd])
    # set_model_mode('eval', [*models_td])

    with torch.no_grad():
        for step, tgt_data in enumerate(tgt_test_loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(), tgt_labels.cuda()
            pred_sd = F.softmax(models_sd(tgt_imgs), dim=1)
            # pred_td = F.softmax(models_td(tgt_imgs), dim=1)
            softmax_sum = pred_sd
            _, final_pred = torch.topk(softmax_sum, 1)
            correct += final_pred.eq(tgt_labels.long().view_as(final_pred)).sum().item()
            total += tgt_labels.size(0)
    # print('当前准确率')
    print('Final Accuracy: {:.2f}%'.format((correct / total) * 100))
    set_model_mode('train', [*models_sd])
    # set_model_mode('train', [*models_td])


def get_mid_mixip_loss(net, src_imgs, tgt_imgs, src_lab, x_sd, total_loss, args, step):
    criterion = nn.CrossEntropyLoss().cuda()
    """pseudo-label"""
    t_loss = total_loss
    tgt_ul_pred = torch.softmax(x_sd, dim=1)
    max_k, k_index = torch.topk(tgt_ul_pred, k=1)
    k_values = k_index.squeeze(1)          # 生成的概率最大的伪标签
    point = (k_index != -1).nonzero()      # 概率最大的伪标签的位置
    copy_tgt_pred = torch.cat([torch.cat((tgt_ul_pred[i][0:k], tgt_ul_pred[i][k+1:])) for (i, _), k in zip((point), k_values)])
    copy_tgt_pred = copy_tgt_pred.view(tgt_ul_pred.size(0), tgt_ul_pred.size(1)-1)

    tgt_pred_var = torch.var(tgt_ul_pred, 1, True)
    copy_tgt_pred_var = torch.var(copy_tgt_pred, 1, True)
    predvar_div_copypredvar = torch.div(tgt_pred_var, copy_tgt_pred_var).unsqueeze(1)
    tgt_ul_mask = predvar_div_copypredvar[:, 0] >= args.threshold
    # print(tgt_ul_mask)

    tgt_ul_img = tgt_imgs[tgt_ul_mask]
    psl_u_2 = k_index.squeeze(1)[tgt_ul_mask]

    """mix_up"""
    alpha = 1
    lam = np.random.beta(alpha, alpha)
    lam = 0.2

    # stream
    if tgt_ul_img.size(0) > 0:
        size_2 = tgt_ul_img.size(0)
        # print('stream 2: {}'.format(size_2))
        s_idx = torch.randperm(src_imgs.size(0))[0:size_2]
        mixed_x = (1 - lam) * src_imgs[s_idx] + lam * tgt_ul_img
        y_a, y_b = src_lab[s_idx], psl_u_2
        out_mix = net(mixed_x)
        loss = (1 - lam) * criterion(out_mix, y_a) + lam * criterion(out_mix, y_b)
        if step == 0:
            print('Mid MixUp Loss: {:.4f}'.format(loss.item()))
        t_loss = t_loss + loss
    return t_loss
    # return loss


def get_penaish_loss(x_sd, total_loss, sp_param_sd, step, args):
    # Self-penalization

    t_loss = total_loss
    tgt_ul_pred = torch.softmax(x_sd, dim=1)
    max_k, k_index = torch.topk(tgt_ul_pred, k=1)
    k_values = k_index.squeeze(1)  # 生成的概率最大的伪标签
    point = (k_index != -1).nonzero()  # 概率最大的伪标签的位置
    copy_tgt_pred = torch.cat(
        [torch.cat((tgt_ul_pred[i][0:k], tgt_ul_pred[i][k + 1:])) for (i, _), k in zip((point), k_values)])
    copy_tgt_pred = copy_tgt_pred.view(tgt_ul_pred.size(0), tgt_ul_pred.size(1) - 1)

    tgt_pred_var = torch.var(tgt_ul_pred, 1, True)
    copy_tgt_pred_var = torch.var(copy_tgt_pred, 1, True)
    predvar_div_copypredvar = torch.div(tgt_pred_var, copy_tgt_pred_var).unsqueeze(1)
    # print(max_k)
    # print(predvar_div_copypredvar)
    # tgt_ul_mask = predvar_div_copypredvar[:, 0] <= args.threshold
    tgt_ul_mask = torch.lt(predvar_div_copypredvar[:, 0], args.threshold)
    tgt_ul_mask = torch.nonzero(tgt_ul_mask).squeeze()
    # print(tgt_ul_mask)


    # tgt_ul_img = tgt_imgs[tgt_ul_mask]
    # psl_u_2 = k_index.squeeze(1)[tgt_ul_mask]

    # sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
    # sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

    if tgt_ul_mask.dim() > 0:
        if tgt_ul_mask.numel() > 0:
            sp_mask = tgt_ul_mask.size(0)
            # print(sp_mask)

            sp_sd_loss = get_sp_loss(x_sd[tgt_ul_mask[:sp_mask]], k_index.squeeze()[tgt_ul_mask[:sp_mask]], sp_param_sd)
            t_loss = t_loss + sp_sd_loss
            if step == 0:
                print('Penalization Loss: {:.4f}'.format(sp_sd_loss.item()))
    return t_loss











