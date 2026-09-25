# coding=utf-8
import time
import torch
import torch.nn as nn
import src.utils as utils


def train_fixbi(args, loaders, optimizers, models_sd, sp_params, losses, epoch):

    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()
    src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    optimizer_sd = optimizers[0]
    sp_param_sd = sp_params[0]
    ce, mse = losses[0], losses[1]

    utils.set_model_mode('train', models=models_sd)

    models_sd = nn.Sequential(*models_sd)

    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)  #数据从CPU移动到GPU的时候，它是异步的
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)

        x_sd = models_sd(tgt_imgs)

        pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args, x_sd)
        fixmix_sd_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, args, step)
        total_loss = fixmix_sd_loss

        # select mix_up
        # if epoch >= args.mix_start:
        #     # mid_mixup_loss=utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, threshold_sd)
        #     # mid_mixup_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, args)
        #     # total_loss = total_loss + mid_mixup_loss
        #     total_loss, mid_mixup_loss = utils.get_mid_mixip_loss(models_sd, src_imgs, tgt_imgs, src_labels, x_sd, total_loss, args)

        # Self-penalization
        if epoch < args.sp_start:
            # total_loss += utils.get_penaish_loss(x_sd, total_loss, sp_param_sd, step, args)

            sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
            sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

            # print(sp_mask_sd.size())
            # print(sp_mask_sd)

            if sp_mask_sd.dim() > 0:
                if sp_mask_sd.numel() > 0:
                    sp_mask = sp_mask_sd.size(0)
                    sp_sd_loss = utils.get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]],
                                                   sp_param_sd)
                    total_loss += sp_sd_loss




        if step == 0:
            print('Fixed MixUp Loss : {:.4f}'.format(fixmix_sd_loss.item()))
            # if epoch > args.mix_start:
            #     print('Mid MixUp Loss: {:.4f}'.format(mid_mixup_loss.item()))
            # if epoch < args.sp_start:
            #     print('Penalization Loss: {:.4f}'.format(sp_sd_loss.item()))

        optimizer_sd.zero_grad()
        total_loss.backward()
        optimizer_sd.step()

    print("Train time: {:.2f}".format(time.time() - start))
