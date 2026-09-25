import matplotlib.pyplot as plt
import torch

import numpy as np
from src.utils_1 import set_model_mode
import torch.nn.functional as F
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


# 画图用的模型文件 models_Ada_noAdain
# 还修改了数据集加载函数，直接注释的
# 画图用的数据集
# -db_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/datasets
# 训练过后的模型
# -baseline_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/profix/save_path

def final_eval2(models_sd, tgt_test_loader):
    total = 0
    correct = 0
    set_model_mode('eval', [*models_sd])
    model = models_sd[0]
    target_layers = [model.layer4[-1]]

    img_ori = cv2.imread('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/Dataset/images/backpacks/hpic_008.jpg')
    img_ori = cv2.resize(img_ori, (224, 224), interpolation=cv2.INTER_CUBIC)
    # print('img_Ori: ', img_ori)
    # print(type(img_ori))
    img_ori_float = img_ori.astype(np.float32) / 255


    # with torch.no_grad():
    for step, tgt_data in enumerate(tgt_test_loader):
        tgt_imgs, tgt_labels = tgt_data
        tgt_imgs, tgt_labels = tgt_imgs.cuda(), tgt_labels.cuda()
        # print(tgt_imgs.shape, tgt_imgs[0].type)
        input_tensor = tgt_imgs
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img_ori_float, grayscale_cam, use_rgb=True)

        plt.imsave('./hpic_008.jpg', visualization)



            # pred_sd = F.softmax(models_sd((tgt_imgs, False)), dim=1)
            # # pred_td = F.softmax(models_td(tgt_imgs), dim=1)
            # softmax_sum = pred_sd
            # _, final_pred = torch.topk(softmax_sum, 1)
            # correct += final_pred.eq(tgt_labels.long().view_as(final_pred)).sum().item()
            # total += tgt_labels.size(0)

    # best_acc = (correct / total) * 100

    # if best_acc >= pre_acc:
    #     print('prs_bestacc: {:.2f}%'.format(pre_acc))
    #     pre_acc = best_acc
    #     save_net('', models_sd, '')
    #     print("save model")
    # print('Final Accuracy: {:.2f}%'.format((correct / total) * 100))
    # set_model_mode('train', [*models_sd])
    # set_model_mode('train', [*models_td])