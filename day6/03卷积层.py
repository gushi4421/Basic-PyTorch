"""
案例:
    演示卷积层的API,用于 提取图像的局部特征,获取: 特征图(Feature Map)

卷积神经网络介绍:
    概述:
        全程叫: Convolutional Neural network (CNN)
    组成:
        卷积层(Convolutional Layer)
            用于提取图像的局部特征,获取: 特征图(Feature Map)
        池化层(Pooling Layer)
            用于降低特征图的尺寸,减少参数量和计算量,防止过拟合
        全连接层(Fully Connected Layer)
            用于 预测结果, 并输出结果
    特征图的计算方式:
        N = (W - F + 2P) / S + 1
        W: 输入图像的宽度(Height)
        F: 卷积核的尺寸(Filter Size)
        P: 填充(Padding)
        S: 步幅(Stride)
        N: 输出特征图的宽度(Height)
"""

import torch
import matplotlib.pyplot as plt


# 1. 定义函数，用于完成图像的加载，卷积，特征图可视化操作
def demo01():
    # 1. 加载RGB真彩图
    img = plt.imread("day6/抱歉，用了你的照片.jpg")  # HWC
    # 2. 打印读取到的图像信息
    print(f"img:{img},shape:{img.shape}")
    # 3. 把图像的形状从HWC -> CHW
    img2 = torch.tensor(img, dtype=torch.float32)
    img2 = img2.permute(2, 0, 1)
    print(f"img2:{img2},shape:{img2.shape}")
    # 4. 增加一个维度,把CHW -> NCHW
    img3 = img2.unsqueeze(dim=0)
    print(f"img3:{img3},shape:{img3.shape}")
    # 5. 创建卷积层对象，提取 特征图
    conv = torch.nn.Conv2d(3, 4, 3, 1, 0)
    # 6. 具体的卷积计算
    conv_img = conv(img3)
    # 7. 打印卷积后的结果
    print(f"conv_img:{conv_img},shape:{conv_img.shape}")
    # 8. 查看提取到的4个特征图
    img4 = conv_img[0]
    print(f"img4:{img4},shpae:{img4.shape}")
    # 9. 把上述的图从 CHW -> HWC
    img5 = img4.permute(1, 2, 0)
    print(f"img5:{img5},shape:{img5.shape}")
    # 10. 可视化第一个通道的特征图
    feature1 = img5[:, :, 1].detach().numpy()  # 第1通道
    plt.imshow(feature1)
    plt.show()


if __name__ == "__main__":
    demo01()
