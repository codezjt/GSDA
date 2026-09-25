# coding=utf-8
import math
import time
import torch
import torch.nn as nn
import src.utils_1 as utils


def train_fixbi(args, loaders, optimizers, models_sd, sp_params, losses, epoch, lam, pre_lam):

    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    print("Lam: {:.2f}".format(lam))
    start = time.time()
    src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    optimizer_sd = optimizers[0]
    # sp_param_sd = sp_params[0]
    ce, mse = losses[0], losses[1]


    utils.set_model_mode('train', models=models_sd)

    models_sd = nn.Sequential(*models_sd)


    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):

        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)  #数据从CPU移动到GPU的时候，它是异步的
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)

        x_sd = models_sd((tgt_imgs, True))

        pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args, x_sd)

        # 原执行顺序
        fixmix_sd_loss = utils.get_fix_sp_loss2(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, step, threshold_sd, lam, pre_lam, mse, args, sp_params)


        # fixmix_sd_loss = utils.get_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, args.lam_sd, args)
        # fixmix_sd_loss = utils.get_fixMatch_loss(models_sd, x_sd, tgt_imgs, sp_params, args)
        # fixmix_sd_loss = utils.get_style_eliminate(models_sd, src_imgs, tgt_imgs, src_labels, mse, lam, epoch, args)
        # fixmix_sd_loss = utils.get_FixMatch_Con_loss(models_sd, src_imgs, tgt_imgs, mse, step)
        # fixmix_sd_loss = utils.get_ST_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, args.lam_sd)
        # fixmix_sd_loss = utils.get_H_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, x_sd)
        # fixmix_sd_loss = utils.get_consistency_loss(models_sd, src_imgs, tgt_imgs, mse, step)
        # fixmix_sd_loss = utils.get_intensifier_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, x_sd, args)
        # fixmix_sd_loss = utils.Cst_1(models_sd, src_imgs, tgt_imgs, src_labels)
        total_loss = fixmix_sd_loss

        # optimizer_sd.zero_grad()
        # total_loss.backward()
        # optimizer_sd.step()

        # select mix_up
        # x_sd = models_sd(tgt_imgs)
        #
        # pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args, x_sd)
        #
        if epoch >= args.style_match:

            mid_mixup_loss = utils.get_Style_Match_loss2(models_sd, x_sd, tgt_imgs, sp_params, args)


            # mid_mixup_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, step, threshold_sd, lam)
            # mid_mixup_loss = utils.get_Allmid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, step, threshold_sd, lam, args, sp_params)
            # mid_mixup_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, args)
            # total_loss = mid_mixup_loss
            if mid_mixup_loss != None:
                total_loss = total_loss + mid_mixup_loss
            else:
                pass
            # total_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, total_loss, args, step)  #selectMixUp
        # # 原执行顺序结束
        # optimizer_sd.zero_grad()
        # if total_loss != None:
        #     total_loss.backward()
        #     optimizer_sd.step()
        # else:
        #     pass
        optimizer_sd.zero_grad()
        total_loss.backward()
        optimizer_sd.step()

        #     # total_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, threshold_sd, total_loss, step) #原MIXup
        #
        # # Self-penalization
        # if epoch < args.sp_start:
        #     total_loss += utils.get_penaish_loss(x_sd, total_loss, sp_params, step, args)
        #
        # #     sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
        # #     sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()
        # #
        # #     # print(sp_mask_sd.size())
        # #     # print(sp_mask_sd)
        # #
        # #     if sp_mask_sd.dim() > 0:
        # #         if sp_mask_sd.numel() > 0:
        # #             sp_mask = sp_mask_sd.size(0)
        # #             sp_sd_loss = utils.get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]],
        # #                                            sp_param_sd)
        # #             total_loss += sp_sd_loss
        #
        # # Consistency Regularization
        # # if epoch > args.con_start:
        # #     consistency_loss = utils.get_consistency_loss(models_sd, src_imgs, tgt_imgs, mse, step)
        # #
        # #     optimizer_sd.zero_grad()
        # #     consistency_loss.backward()
        # #     optimizer_sd.step()
        #

        # # 原执行
        if step == 0:
            if fixmix_sd_loss != None:
                print('Total Loss : {:.4f}'.format(fixmix_sd_loss.item()))
        # # 原执行结束
            # if epoch > args.mix_start:
            #     print('Mid MixUp Loss: {:.4f}'.format(mid_mixup_loss.item()))
            # if epoch < args.sp_start:
            #     print('Penalization Loss: {:.4f}'.format(sp_sd_loss.item()))

        # optimizer_sd.zero_grad()
        # total_loss.backward()
        # optimizer_sd.step()
    print("Train time: {:.2f}".format(time.time() - start))
