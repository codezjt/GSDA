import matplotlib.pyplot as plt
import torch

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification


def tsne_visil(features, labels, class_num):


    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 绘制 t-SNE 降维后的结果，每个类别用不同的颜色表示
    plt.figure(figsize=(10, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, class_num))
    for i in range(class_num):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], color=colors[i], label=str(i), s=5)
    # plt.title('t-SNE Visualization of Last Hidden Layer Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class')
    plt.savefig("Dtsne1.jpg")
    plt.show()


