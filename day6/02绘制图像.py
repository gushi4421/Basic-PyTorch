"""
案例:
    演示基础的图像操作

图像分类:
    二绘图:     1通道,每个像素点由0,1组成
    灰度图:     1通道,每个像素点的范围:[0,255]
    索引图:     1通道,每个像素点的范围:[0,255],像素点表示颜色表的索引
    RGB真彩图:

涉及到的API:
    imshow()    基于HWC,展示图像
    imread()    基于图像,获取HWC
    imsave()    基于HWC,保存图片
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


# 1. 定义函数:绘制: 全黑，全白图
def demo01():
    # 1. 定义全黑图片: 像素点越接近0越黑,越接近255越白
    # HWC: H: 高度，W: 宽度,C: 通道
    img1 = np.zeros((200, 200, 3))

    # 绘制图片
    plt.imshow(img1)
    plt.show()

    # 2. 绘制全白图片
    img2 = torch.full(size=(200, 200, 3), full_size=255)
    plt.imshow(img2)
    plt.show()


# 2. 定义函数: 加载图片
def demo02():
    # 1. 读取图片
    img1 = plt.imread("day6/抱歉，用了你的照片.jpg")
    print(f"img1:{img1}")
    # 2. 保存图像
    plt.imsave("day6/抱歉，用了你的照片_copy.png", img1)
    # 3. 展示图像
    plt.imshow(img1)
    plt.show()


if __name__ == "__main__":
    # demo01()
    demo02()
