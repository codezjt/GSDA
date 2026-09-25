# coding=utf-8
import copy
import time
import os
import random
import tensorflow as tf


import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from utils.tsneview import tsne_visil
from utils.utils import AverageMeter


# import network.models_Ada_noAdain as models
import network.models_Ada as models               # gard_cam

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_info():
    resnet_type = 50     # ResNet50
    # resnet_type = 101   # ResNet101
    # num_classes = 65  # office-home类别数
    num_classes = 31  # office-31类别数
    # num_classes = 12  # VisDA类别数
    return num_classes, resnet_type


def get_net_info(num_classes):

    # net = models.ResNet101().encoder.cuda()  # 加载ResNet101
    net = models.ResNet50().encoder.cuda()  #该容器通过在批处理维度中分组，将输入分割到指定的设备上，
    classifier = nn.Linear(256, num_classes).cuda()
    head = models.Head().cuda()


    return net, head, classifier


def get_train_info():
    lr = 0.001
    l2_decay = 5e-4
    momentum = 0.95
    nesterov = False
    return lr, l2_decay, momentum, nesterov



def load_net(args, net, head, classifier):
    print("Load pre-trained baseline model !")
    save_folder = args.baseline_path
    net.load_state_dict(torch.load(save_folder + '/net.pt'), strict=False)
    # net.load_state_dict(torch.load('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/resnet50-19c8e357.pth'), strict=False)
    head.load_state_dict(torch.load(save_folder + '/head.pt'), strict=False)
    classifier.load_state_dict(torch.load(save_folder + '/classifier.pt'), strict=False)
    return net, head, classifier


def save_net(args, models, type):
    save_folder = '/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/FixBi/save_path/A_D'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    net, head, classifier = models[0], models[1], models[2]

    torch.save(net.state_dict(), save_folder + '/' + 'net' + '.pt')
    torch.save(head.state_dict(), save_folder + '/' + 'head' + '.pt')
    torch.save(classifier.state_dict(), save_folder + '/' + 'classifier' + '.pt')


def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()

def test(models, target_test_loader):
    models.eval()
    test_loss = AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            s_output = models((data, False))
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    test_acc, test_loss = acc, test_loss.avg
    info = 'test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
    print(info)


def evaluate(models, loader):
    p_adain = True
    start = time.time()
    total = 0
    correct = 0
    set_model_mode('eval', [models])
    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
            tgt_preds = models((tgt_imgs, False))
            pred = tgt_preds.argmax(dim=1, keepdim=True)
            correct += pred.eq(tgt_labels.long().view_as(pred)).sum().item()
            total += tgt_labels.size(0)

    print('Accuracy: {:.2f}%'.format((correct / total) * 100))
    print("Eval time: {:.2f}".format(time.time() - start))
    set_model_mode('train', [models])

def getTsne(models, loader):
    p_adain = True
    start = time.time()

    set_model_mode('eval', [models])
    modelpre = models[0]
    count = 0
    # fanal_feature = tf.constant([], dtype=tf.float32)
    fanal_feature = None
    fanal_label = None
    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            count += 1
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
            features = modelpre((tgt_imgs, False))
            # fanal_feature = torch.sum(features, dim=0)
            if fanal_feature is None:
                fanal_feature = features
                fanal_label = tgt_labels
            else:
                # print(fanal_feature.size(), features.size())
                fanal_feature = torch.cat((fanal_feature, features), dim=0)
                fanal_label = torch.cat((fanal_label, tgt_labels), dim=0)
                fanal_label = fanal_label.squeeze()
    print(fanal_feature.size(), fanal_label.size())
    last_features = fanal_feature.cpu().numpy()
    last_label = fanal_label.cpu().numpy()
    print(fanal_feature.size(), fanal_label.size())
    tsne_visil(last_features, last_label, 31)



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


def H_mixup_criterion_hard(pred, y_a, y_b, lam):
    C = 65
    N = y_a.size(0)
    labels_a = torch.full(size=(N,C), fill_value=0)
    labels_b = torch.full(size=(N,C), fill_value=0)
    labels_a.scatter_(dim=1, index=torch.unsqueeze(y_a.cpu(), dim=1), value=1)
    labels_b.scatter_(dim=1, index=torch.unsqueeze(y_b.cpu(), dim=1), value=1)

    log_prob = torch.nn.functional.log_softmax(pred, dim=1)

    lam = lam.unsqueeze(-1)
    labels_a = labels_a.cuda()
    labels_b = labels_b.cuda()
    lam = lam.cuda()
    loss = -torch.sum(log_prob * (lam * labels_a + (1-lam) * labels_b)) / N
    return loss


def get_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio, args):
    mixed_x1 = ratio * src_imgs + (1 - ratio) * tgt_imgs
    mixed_x = net((mixed_x1, True))
    loss = mixup_criterion_hard(mixed_x, src_labels.detach(), tgt_pseudo.detach(), ratio)
    return loss

def get_sp_loss(input, target, temp):
    criterion = nn.NLLLoss(reduction='none').cuda()
    loss = torch.mul(criterion(torch.log(1 - F.softmax(input / temp, dim=1)), target.detach()), 1).mean()
    return loss


def get_fixMatch_loss(net, x_sd, tgt_imgs, sp_params, args):
    # criterion = nn.CrossEntropyLoss().cuda()
    roi_out = net((tgt_imgs, False))
    roi_label, _, th = get_target_preds(args, roi_out)
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)
    # enh_out = net((tgt_imgs, True))
    pseudo_label = torch.softmax(roi_out.detach(), dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(0.6).float()
    loss = (F.cross_entropy(x_sd, targets_u, reduction='none') * mask).mean()
    # # # fix_Penalization
    # sp_mask = torch.lt(top_prob_sd, threshold_sd)
    # sp_mask = torch.nonzero(sp_mask).squeeze()
    # if sp_mask.dim() > 0:
    #     if sp_mask.numel() > 0:
    #         sp_mask_size = sp_mask.size(0)
    #         sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
    #         if sp_loss is not None:
    #             loss += sp_loss
    #         else:
    #             pass
    # # loss = criterion(x_sd, targets_u)
    return loss

def get_fixMatch_loss_CE(net, x_sd, tgt_imgs, sp_params, args):
    criterion = nn.CrossEntropyLoss().cuda()
    roi_out = net((tgt_imgs, False))
    roi_label, _, th = get_target_preds(args, roi_out.detach())
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)
    u_2_prob = torch.softmax(roi_out.detach(), dim=1)
    u_2_pred = u_2_prob.max(1)
    u_2_mask = u_2_pred[0] >= th

    x_sd_mask = x_sd[u_2_mask]
    psl_u_mask = u_2_pred[1][u_2_mask]
    loss = criterion(x_sd_mask, psl_u_mask)
    return loss

def get_fix_Penalization(net, x_sd, tgt_imgs, sp_params, args):
    roi_out = net((tgt_imgs, False))
    _, _, th = get_target_preds(args, roi_out)
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)
    sp_mask = torch.lt(top_prob_sd, th)
    sp_mask = torch.nonzero(sp_mask).squeeze()
    if sp_mask.dim() > 0:
        if sp_mask.numel() > 0:
            sp_mask_size = sp_mask.size(0)
            sp_loss = get_sp_loss((x_sd)[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
            return sp_loss



#
# def get_FixMatch_Con_loss(net, src_imgs, tgt_imgs, mse, step):
#     SMix_imgs = 0.70 * src_imgs + 0.30 * tgt_imgs
#     # TMix_imgs = 0.40 * src_imgs + 0.60 * tgt_imgs
#     out_sd = net((SMix_imgs, True))
#     with torch.no_grad():
#         out_td = net((SMix_imgs, False))
#     cr_loss = mse(out_sd, out_td)
#     # if step == 0:
#     #     print('Consistency Loss: {:.4f}'.format(cr_loss.item()))
#     return cr_loss

# def get_fixMatch_loss(net, x_sd, tgt_imgs, pseudo_sd, args):
#
#     roi_out = net((tgt_imgs, False))
#     _, _, th = get_target_preds(args, roi_out)
#     # enh_out = net((tgt_imgs, True))
#     pseudo_label = torch.softmax(roi_out.detach(), dim=-1)
#     max_probs, targets_u = torch.max(pseudo_label, dim=-1)
#     mask = max_probs.ge(th).float()
#     p_label, _, _ = get_target_preds(args, roi_out)
#     loss = (F.cross_entropy(x_sd, pseudo_sd, reduction='none')).mean()
#     return loss


# def get_fixMatch_loss(net, x_sd, tgt_imgs, args):
#     criterion = nn.CrossEntropyLoss().cuda()
#     with torch.no_grad():
#         roi_out = net((tgt_imgs, True))
#     # _, _, th = get_target_preds(args, roi_out)
#     # # enh_out = net((tgt_imgs, True))
#     # pseudo_label = torch.softmax(roi_out.detach(), dim=-1)
#     # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
#     # mask = max_probs.ge(th).float()
#     # # loss = (F.cross_entropy(x_sd, targets_u, reduction='none') * mask).mean()
#     # loss = criterion(x_sd, targets_u)
#     return None


def get_ST_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio):
    criterion = nn.CrossEntropyLoss().cuda()
    src_imgs1, src_imgs2 = torch.chunk(src_imgs, 2, dim=0)
    tgt_imgs1, tgt_imgs2 = torch.chunk(tgt_imgs, 2, dim=0)
    mix_imgs1 = torch.cat((src_imgs1, tgt_imgs1), dim=0)
    mix_imgs2 = torch.cat((src_imgs2, tgt_imgs2), dim=0)

    src_labels1, src_labels2 = torch.chunk(src_labels, 2, dim=0)
    tgt_pseudo1, tgt_pseudo2 = torch.chunk(tgt_pseudo, 2, dim=0)
    mix_label1 = torch.cat((src_labels1, tgt_pseudo1), dim=0)
    mix_label2 = torch.cat((src_labels2, tgt_pseudo2), dim=0)

    for step, (imgs, labels) in enumerate(zip((mix_imgs1, mix_imgs2), (mix_label1, mix_label2))):
        mixed_x = net(imgs)
        loss = criterion(mixed_x, labels)
        if step == 0:
            t_loss = loss
        else:
            t_loss += loss
    return t_loss


def get_H_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, x_sd):
    pred = torch.softmax(x_sd, dim=1)
    score = pred.max(1)
    ratio = score[0]
    un_ratio = ratio
    ratio = ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # print(ratio)
    mixed_x = (1 - ratio) * src_imgs + ratio * tgt_imgs
    mixed_x = net(mixed_x)
    loss = H_mixup_criterion_hard(mixed_x, src_labels.detach(), tgt_pseudo.detach(), 1-un_ratio)
    return loss

def get_intensifier_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, x_sd, args):
    criterion = nn.CrossEntropyLoss().cuda()
    SMix_imgs = 0.60 * src_imgs + 0.40 * tgt_imgs
    TMix_imgs = 0.40 * src_imgs + 0.60 * tgt_imgs
    out_sd = net(SMix_imgs)
    with torch.no_grad():
        out_td = net(TMix_imgs)
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, out_td)
    cr_loss = criterion(out_sd, pseudo_sd)
    # cr_loss = mse(out_sd, out_td)
    # if step == 0:
    #     print('Consistency Loss: {:.4f}'.format(cr_loss.item()))
    return cr_loss

# pre_acc = 0
def final_eval(models_sd, tgt_test_loader):
    # print("仅测试*************************")
    # models_sd[0].load_state_dict(torch.load('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/DeepDA/save_floder/net.pt'), strict=False)
    # models_sd[1].load_state_dict(torch.load('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/DeepDA/save_floder/head.pt'), strict=False)
    # models_sd[2].load_state_dict(torch.load('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/DeepDA/save_floder/classifier.pt'), strict=False)
    total = 0
    correct = 0
    p_adain = True
    set_model_mode('eval', [*models_sd])
    # set_model_mode('eval', [*models_td])
    # global pre_acc
    with torch.no_grad():
        for step, tgt_data in enumerate(tgt_test_loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(), tgt_labels.cuda()
            pred_sd = F.softmax(models_sd((tgt_imgs, False)), dim=1)
            # pred_td = F.softmax(models_td(tgt_imgs), dim=1)
            softmax_sum = pred_sd
            _, final_pred = torch.topk(softmax_sum, 1)
            correct += final_pred.eq(tgt_labels.long().view_as(final_pred)).sum().item()
            total += tgt_labels.size(0)

    # best_acc = (correct / total) * 100

    # if best_acc >= pre_acc:
    #     print('prs_bestacc: {:.2f}%'.format(pre_acc))
    #     pre_acc = best_acc
    #     save_net('', models_sd, '')
    #     print("save model")
    print('Final Accuracy: {:.2f}%'.format((correct / total) * 100))
    set_model_mode('train', [*models_sd])
    # set_model_mode('train', [*models_td])


def get_mid_mixip_loss(net, src_imgs, tgt_imgs, src_lab, x_sd, step, th, lam):
    criterion = nn.CrossEntropyLoss().cuda()

    """pseudo-label"""
    im_data_tu = tgt_imgs
    im_data_s = src_imgs
    gt_labels_s = src_lab

    u_2_prob = torch.softmax(x_sd, dim=1)
    u_2_pred = u_2_prob.max(1)

    u_2_mask = u_2_pred[0] >= th

    im_u_2 = im_data_tu[u_2_mask]
    psl_u_2 = u_2_pred[1][u_2_mask]

    """mix_up"""
    alpha = 1
    lam = np.random.beta(alpha, alpha)
    while(lam > 0.7) | (lam < 0.3):
        lam = np.random.beta(alpha, alpha)

    # stream
    if im_u_2.size(0) > 0:
        size_2 = im_u_2.size(0)
        # print('stream 2: {}'.format(size_2))
        s_idx = torch.randperm(im_data_s.size(0))[0:size_2]
        mixed_x = (1 - lam) * im_data_s[s_idx] + lam * im_u_2
        y_a, y_b = gt_labels_s[s_idx], psl_u_2
        out_mix = net((mixed_x, True))
        loss = (1 - lam) * criterion(out_mix, y_a) + lam * criterion(out_mix, y_b)
        if step == 0:
            print('Mid MixUp Loss: {:.4f}'.format(loss.item()))
        return loss
    #     total_loss += loss
    #     return total_loss
    # else:
    #     return total_loss

def get_Allmid_mixip_loss(net, src_imgs, tgt_imgs, src_lab, x_sd, step, th, lam, args, sp_params):
    criterion = nn.CrossEntropyLoss().cuda()

    """pseudo-label"""
    im_data_tu = tgt_imgs
    # im_data_s = src_imgs
    # gt_labels_s = src_lab

    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)

    # u_2_prob = torch.softmax(x_sd, dim=1)
    # u_2_pred = u_2_prob.max(1)    # x_sd的伪标签
    #
    # u_2_mask = u_2_pred[0] >= th

    # im_u_2 = im_data_tu[u_2_mask]
    # psl_u_2 = u_2_pred[1][u_2_mask]

    """mix_up"""
    alpha = 1
    lam = np.random.beta(alpha, alpha)
    while(lam > 0.7) | (lam < 0.3):
        lam = np.random.beta(alpha, alpha)
    mixed_x = (1-lam) * src_imgs + lam * tgt_imgs
    out_mix = net((mixed_x, True))
    loss = mixup_criterion_hard(out_mix, src_lab.detach(), pseudo_sd.detach(), 1-lam)
    # Fix Penalization
    sp_mask = torch.lt(top_prob_sd, th)
    sp_mask = torch.nonzero(sp_mask).squeeze()
    if sp_mask.dim() > 0:
        if sp_mask.numel() > 0:
            sp_mask_size = sp_mask.size(0)
            sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
            if sp_loss is not None:
                loss += sp_loss
            else:
                pass
    if step == 0:
        print('Mid MixUp Loss: {:.4f}'.format(loss.item()))
    return loss



    # # stream
    # if im_u_2.size(0) > 0:
    #     size_2 = im_u_2.size(0)
    #     # print('stream 2: {}'.format(size_2))
    #     s_idx = torch.randperm(im_data_s.size(0))[0:size_2]
    #     mixed_x = (1 - lam) * im_data_s[s_idx] + lam * im_u_2
    #     y_a, y_b = gt_labels_s[s_idx], psl_u_2
    #     out_mix = net((mixed_x, True))
    # #     loss = (1 - lam) * criterion(out_mix, y_a) + lam * criterion(out_mix, y_b)
    # if step == 0:
    #     print('Mid MixUp Loss: {:.4f}'.format(loss.item()))
    #     return loss

# consistency regularization
def get_consistency_loss(net, src_imgs, tgt_imgs, mse, step):
    SMix_imgs = 0.60 * src_imgs + 0.40 * tgt_imgs
    TMix_imgs = 0.40 * src_imgs + 0.60 * tgt_imgs
    out_sd = net((SMix_imgs, True))
    with torch.no_grad():
        out_td = net((TMix_imgs, True))
    cr_loss = mse(out_sd, out_td)
    # if step == 0:
    #     print('Consistency Loss: {:.4f}'.format(cr_loss.item()))
    return cr_loss


def get_penaish_loss(x_sd, top_prob_sd, threshold_sd, pseudo_sd, sp_param_sd):
    # Self-penalization
    sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
    sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

    if sp_mask_sd.dim() > 0:
        if sp_mask_sd.numel() > 0:
            sp_mask = sp_mask_sd.size(0)
            sp_sd_loss = get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]], sp_param_sd)
            return sp_sd_loss
        else:
            return None

def Cst_1(models_sd, src_imgs, tgt_imgs, src_labels):
    criterion = nn.CrossEntropyLoss().cuda()
    # head_td, classifier_td = copy.deepcopy(models_sd[1]), copy.deepcopy(models_sd[2])

    head_td, classifier_td = models_sd[1], models_sd[2]
    net = nn.Sequential(*[models_sd[0]])
    nets_c = nn.Sequential(*[models_sd[1], models_sd[2]])
    nett_c = nn.Sequential(*[head_td, classifier_td])

    # netoo =nn.Sequential(*models_sd)
    # oo = models_sd(src_imgs)
    # pp, _, _, = get_target_preds(args, oo)
    # ll = criterion(pp, src_labels)


    #
    S_out = net(src_imgs)
    T_out = net(tgt_imgs)

    S_nets_c = nets_c(S_out)
    T_nets_c = nets_c(T_out)

    # S_nett_c = nett_c(S_out)
    # T_nett_c = nett_c(T_out)

    y_s = S_nets_c
    y_t = T_nets_c
    f_s = S_out
    f_t = T_out
    y_t_u = T_nets_c
    labels_s = src_labels

    # generate target pseudo-labels
    max_prob, pred_u = torch.max(F.softmax(y_t, dim=1), dim=-1)
    Lu = (F.cross_entropy(y_t_u, pred_u,
                          reduction='none') * max_prob.ge(0.97).float().detach()).mean()

    # compute cst
    target_data_train_r = f_t
    target_data_train_r = target_data_train_r / (
        torch.norm(target_data_train_r, dim=-1).reshape(target_data_train_r.shape[0], 1))
    target_data_test_r = f_s
    target_data_test_r = target_data_test_r / (
        torch.norm(target_data_test_r, dim=-1).reshape(target_data_test_r.shape[0], 1))
    target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                                0.99999999)
    target_kernel_r = target_gram_r
    test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0=1, dim1=0)), -0.99999999,
                              0.99999999)
    test_kernel_r = test_gram_r
    target_train_label_r = torch.nn.functional.one_hot(pred_u, 65) - 1 / float(65)
    target_test_pred_r = test_kernel_r.mm(
        torch.inverse(target_kernel_r + 0.001 * torch.eye(32).cuda())).mm(target_train_label_r)
    reverse_loss = nn.MSELoss()(target_test_pred_r,
                                torch.nn.functional.one_hot(labels_s, 65) - 1 / float(65))

    cls_loss = F.cross_entropy(y_s, labels_s)
    ts_loss = TsallisEntropy(temperature=2.0, alpha=1.9)   #####
    transfer_loss = ts_loss(y_t)

    if Lu != 0:
        loss = cls_loss + transfer_loss * 0.08 + reverse_loss * 0.5 + Lu * 0.5
    else:
        loss = cls_loss + transfer_loss * 0.08 + reverse_loss * 0.5

    return loss


class TsallisEntropy(nn.Module):

    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape

        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)

        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

        return 1 / (self.alpha - 1) * torch.sum(
            (1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1)))


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

def get_style_eliminate(net, src_imgs, tgt_imgs, src_labels, mse, lam, epoch, args):
    criterion = nn.CrossEntropyLoss().cuda()
    src_tgt_imgs = torch.cat((src_imgs, tgt_imgs), dim=0)
    base_net_out = net((src_tgt_imgs, True))
    src_net_out, tgt_net_out = base_net_out.chunk(2, dim=0)      # 风格置换后源域目标域的输出
    # src_loss = criterion(src_net_out, src_labels)
    with torch.no_grad():
        tgt_net_out_nogard = net((tgt_imgs, False))
    pseudo, top_prob, threshold = get_target_preds(args, tgt_net_out)
    pseudo_labels = torch.softmax(tgt_net_out_nogard, dim=-1)
    max_prods, targets_u = torch.max(pseudo_labels, dim=-1)
    mask = max_prods.ge(threshold).float()
    tgt_loss = (F.cross_entropy(tgt_net_out, targets_u, reduction='none') * mask).mean()
    # tgt_mse_loss = mse(tgt_net_out, tgt_net_out_nogard)

    loss = tgt_loss


    if epoch >= args.mix_start:
        """pseudo-label"""
        u_2_prob = torch.softmax(tgt_net_out, dim=1)
        u_2_pred = u_2_prob.max(1)

        u_2_mask = u_2_pred[0] >= threshold

        im_u_2 = tgt_imgs[u_2_mask]
        psl_u_2 = u_2_pred[1][u_2_mask]

        """mix_up"""
        # alpha = 1F
        # lam = np.random.beta(alpha, alpha)
        # while(lam > 0.7) | (lam < 0.3):
        #     lam = np.random.beta(alpha, alpha)

        # stream
        if im_u_2.size(0) > 0:
            size_2 = im_u_2.size(0)
            # print('stream 2: {}'.format(size_2))
            s_idx = torch.randperm(src_imgs.size(0))[0:size_2]
            mixed_x = (1 - lam) * src_imgs[s_idx] + lam * im_u_2
            y_a, y_b = src_labels[s_idx], psl_u_2
            out_mix = net((mixed_x, True))
            mid_loss = (1 - lam) * criterion(out_mix, y_a) + lam * criterion(out_mix, y_b)
            if mid_loss != None:
                loss += mid_loss
            # if step == 0:
            #     print("Src En Loss: {:.4f}".format(src_loss.item()))
            #     print("Tgt Mse Loss: {:.4f}".format(tgt_mse_loss.item()))
            #     print('Mid MixUp Loss: {:.4f}'.format(mid_loss.item()))

    return loss


# def get_style_loss(net, src_imgs, tgt_imgs, src_labels, mse, lam, step, args):
#     criterion = nn.CrossEntropyLoss().cuda()
#     src_tgt_imgs = torch.cat((src_imgs, tgt_imgs), dim=0)
#     base_net_out = net((src_tgt_imgs, True))
#     src_net_out, tgt_net_out = base_net_out.chunk(2, dim=0)
#     src_loss = criterion(src_net_out, src_labels)               #源域风格置换到目标域后，目标域的损失


def get_fix_sp_loss2(net, src_imgs, tgt_imgs, src_lab, x_sd, step, th, lam, pre_lam, mse, args, sp_params):
    criterion = nn.CrossEntropyLoss().cuda()

    """pseudo-label"""
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)
    """mix_up"""
    # alpha = 1
    # lam = np.random.beta(alpha, alpha)
    # while(lam > 1) | (lam < 0):
    #     lam = np.random.beta(alpha, alpha)
    # lam = 0.5
    mixed_x = (1-lam) * src_imgs + lam * tgt_imgs
    mixed_src_x = torch.cat((mixed_x, src_imgs), dim=0)
    out_mix_src = net((mixed_src_x, True))
    out_mix, out_src = torch.chunk(out_mix_src, 2, dim=0)
    # # 计算混合输出方差
    # out_mix_pred = torch.softmax(out_mix, dim=1)
    # tgt_ul_pred = out_mix_pred
    # max_k, k_index = torch.topk(tgt_ul_pred, k=1)
    # k_values = k_index.squeeze(1)  # 生成的概率最大的伪标签
    # point = (k_index != -1).nonzero()  # 概率最大的伪标签的位置
    # copy_tgt_pred = torch.cat(
    #     [torch.cat((tgt_ul_pred[i][0:k], tgt_ul_pred[i][k + 1:])) for (i, _), k in zip((point), k_values)])
    # copy_tgt_pred = copy_tgt_pred.view(tgt_ul_pred.size(0), tgt_ul_pred.size(1) - 1)
    #
    # # tgt_pred_var = torch.var(tgt_ul_pred, 1, True)  # 预测概率总方差
    # # total_var[0] = total_var[0] + tgt_pred_var
    # copy_tgt_pred_var = torch.var(copy_tgt_pred, 1, True)  # 去除最大值方差
    # tgt_pred_var = torch.var(out_mix_pred, 1, True)  # 预测概率总方差
    # tot_1_var[0] = tot_1_var[0] + copy_tgt_pred_var
    # total_var[0] = total_var[0] + tgt_pred_var
    #
    # # 计算混合方差输出结束
    mix_loss = mixup_criterion_hard(out_mix, src_lab.detach(), pseudo_sd.detach(), 1 - lam)
    src_loss = criterion(out_src, src_lab)
    loss = mix_loss + src_loss

    if step == 0:
        print('Mid Con Loss: {:.4f}'.format(mix_loss.item()))
    return loss

def get_fix_sp_loss(net, src_imgs, tgt_imgs, src_lab, x_sd, step, th, lam, pre_lam, mse, args, sp_params):
    criterion = nn.CrossEntropyLoss().cuda()

    """pseudo-label"""
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)
    """mix_up"""
    # alpha = 1
    # lam = np.random.beta(alpha, alpha)
    # while(lam > 0.7) | (lam < 0.3):
    #     lam = np.random.beta(alpha, alpha)
    lam = 0.5
    mixed_x = (1-lam) * src_imgs + lam * tgt_imgs
    out_mix = net((mixed_x, True))
    mix_loss = mixup_criterion_hard(out_mix, src_lab.detach(), pseudo_sd.detach(), 1 - lam)
    loss = mix_loss
    # if lam != pre_lam:
    #     pre_mixed_x = (1-pre_lam) * src_imgs + pre_lam * tgt_imgs
    #     with torch.no_grad():
    #         pre_out_mix = net((pre_mixed_x, True))
    #     con_loss = mse(out_mix, pre_out_mix)
    #     if step == 0:
    #         print('Mid MixUp Loss: {:.4f}'.format(con_loss.item()))
    #     loss += con_loss

    # Fix Penalization
    # sp_mask = torch.lt(top_prob_sd, th)
    # sp_mask = torch.nonzero(sp_mask).squeeze()
    # if sp_mask.dim() > 0:
    #     if sp_mask.numel() > 0:
    #         sp_mask_size = sp_mask.size(0)
    #         sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
    #         if sp_loss is not None:
    #             loss += sp_loss
    #         else:
    #             pass
    if step == 0:
        print('Mid Con Loss: {:.4f}'.format(mix_loss.item()))
    return loss

def get_Style_Match_loss2(net, x_sd, tgt_imgs, sp_params, args):
    # criterion = nn.CrossEntropyLoss().cuda()
    numclass, _ = get_data_info()
    with torch.no_grad():
        roi_out = net((tgt_imgs, False))
    # roi_label, _, th = get_target_preds(args, roi_out)

    # *******************
    # # _, _, th = get_target_preds(args, roi_out)
    # pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, roi_out)
    # tgt_ul_mask = torch.lt(top_prob_sd, 0.6)
    # # tgt_ul_mask = torch.nonzero(sp_mask).squeeze()
    # pseudo_label = torch.softmax(roi_out, dim=-1)
    # max_probs, targets_u = torch.max(pseudo_label, dim=-1)

    # ********************
    pseudo_label = torch.softmax(roi_out, dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)

    tgt_ul_pred = torch.softmax(roi_out, dim=1)
    max_k, k_index = torch.topk(tgt_ul_pred, k=1)
    k_values = k_index.squeeze(1)          # 生成的概率最大的伪标签
    point = (k_index != -1).nonzero()      # 概率最大的伪标签的位置
    copy_tgt_pred = torch.cat([torch.cat((tgt_ul_pred[i][0:k], tgt_ul_pred[i][k+1:])) for (i, _), k in zip((point), k_values)])
    copy_tgt_pred = copy_tgt_pred.view(tgt_ul_pred.size(0), tgt_ul_pred.size(1)-1)

    tgt_pred_var = torch.var(tgt_ul_pred, 1, True)        # 预测概率总方差
    copy_tgt_pred_var = torch.var(copy_tgt_pred, 1, True)    # 去除最大值方差
    predvar_div_copypredvar = torch.div(tgt_pred_var, copy_tgt_pred_var).unsqueeze(1)
    tgt_ul_mask = predvar_div_copypredvar[:, 0] >= args.threshold
    # *********************
    # tgt_ul_one_hot_mask = F.one_hot(tgt_ul_mask, numclass)

    # mask = max_probs.ge(0.6).float()
    loss = (F.cross_entropy(x_sd, targets_u, reduction='none') * tgt_ul_mask).mean()
    # # # fix_Penalization
    # sp_mask = torch.lt(top_prob_sd, threshold_sd)
    # sp_mask = torch.nonzero(sp_mask).squeeze()
    # if sp_mask.dim() > 0:
    #     if sp_mask.numel() > 0:
    #         sp_mask_size = sp_mask.size(0)
    #         sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
    #         if sp_loss is not None:
    #             loss += sp_loss
    #         else:
    #             pass
    # # loss = criterion(x_sd, targets_u)

    # Fix Penalization
    # sp_mask = torch.lt(top_prob_sd, th)
    # sp_mask = torch.nonzero(sp_mask).squeeze()
    # if sp_mask.dim() > 0:
    #     if sp_mask.numel() > 0:
    #         sp_mask_size = sp_mask.size(0)
    #         sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
    #         if sp_loss is not None:
    #             loss += sp_loss
    #         else:
    #             pass
    return loss

def get_Style_Match_loss(net, x_sd, tgt_imgs, sp_params, args):
    # criterion = nn.CrossEntropyLoss().cuda()
    roi_out = net((tgt_imgs, False))
    roi_label, _, th = get_target_preds(args, roi_out)
    pseudo_sd, top_prob_sd, threshold_sd = get_target_preds(args, x_sd)
    # enh_out = net((tgt_imgs, True))
    pseudo_label = torch.softmax(roi_out.detach(), dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(0.6).float()
    loss = (F.cross_entropy(x_sd, targets_u, reduction='none') * mask).mean()
    # # # fix_Penalization
    # sp_mask = torch.lt(top_prob_sd, threshold_sd)
    # sp_mask = torch.nonzero(sp_mask).squeeze()
    # if sp_mask.dim() > 0:
    #     if sp_mask.numel() > 0:
    #         sp_mask_size = sp_mask.size(0)
    #         sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
    #         if sp_loss is not None:
    #             loss += sp_loss
    #         else:
    #             pass
    # # loss = criterion(x_sd, targets_u)

    # Fix Penalization
    # sp_mask = torch.lt(top_prob_sd, th)
    # sp_mask = torch.nonzero(sp_mask).squeeze()
    # if sp_mask.dim() > 0:
    #     if sp_mask.numel() > 0:
    #         sp_mask_size = sp_mask.size(0)
    #         sp_loss = get_sp_loss(x_sd[sp_mask[:sp_mask_size]], pseudo_sd[sp_mask[:sp_mask_size]], sp_params)
    #         if sp_loss is not None:
    #             loss += sp_loss
    #         else:
    #             pass
    return loss
























