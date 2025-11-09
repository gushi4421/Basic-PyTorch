"""

正则化的作用:
    缓解模型的过拟合情况

正则化的方式:
    L1正则化: 权重可以变为0,相当于:降维
    L2正则化: 权重可以无限接近于0
    DropOut:  随机失活,每批次样本训练时,随机让一部分神经元死亡
    BN(批量归一化):

批量归一化:
    思路:先对数据做标准化(会丢失一些信息)，然后再对数据做 缩放(λ，理解为:W权重)和 平移(β,理解为:b偏置)，再找补回一些信息。

    应用场景:
        批量归一化在计算机视觉领域使用较多

        BatchNorm1d:主要成用于全连接层或处理一维数据的网络，例如文本处理、它接收形状为(N,num_features) 的张量作为输入
        BatchNorm2d:主要应用于卷积神经网络，处理二维图像数据或特征图、它接收形状为(N,C,H,W)的张量作为输入
        BatchNorm3d:主要用于三维卷积神经网络(3 CNN)，处理三维数据，例如视频或医学图像。它接收形状为(N,C,D,H,W)的张量作为输入
"""

import torch


# 1. 定义函数,演示: 随机失活(dropout)
def demo01():
    # 1. 创建隐藏层输出结果
    t1 = torch.randint(0, 10, size=(1, 4)).float()

    # 2. 进行下一层 加权求和 和 激活函数计算
    # 2.1 创建全连接层(充当线性层)
    # 参1: 输入特征维数,参2: 输出特征维数
    linear1 = torch.nn.Linear(4, 5)

    # 2.2 加权求和
    l1 = linear1(t1)
    print(f"l1:{l1}")

    # 2.3 激活函数
    output = torch.relu(l1)
    print(f"output:{output}")

    # 3. 对激活值进行随机失活
    dropout = torch.nn.Dropout(p=0.4)
    # 具体的失活动作
    d1 = dropout(output)
    print(f"d1(随机失活后的数据):{d1}")  # 未被失活的进行缩放,变为 1 / (1 - p)


# 2. 定义函数,演示: 批量归一化 处理二维数据
def demo02():
    # 1. 创建图像样本数据
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print(f"input_2d:{input_2d}")

    # 2. 创建批量归一化层(BN层)
    # 参1: 输入特征数 = 图片的通道数
    # 参2: 噪声值(小常数),默认为1e-5
    # 参3: 动量值,用于计算移动平均统计量的 动量值
    # 参4: 表示使用可学习的变换参数(λ,β) 对归一化(标准化)后的数据进行 缩放和平移
    bn2d = torch.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)

    # 3. 对数据进行 批量归一化操作
    output_2d = bn2d(input_2d)
    print(f"ouput_2d:{output_2d}")


# 3. 定义函数,演示: 批量归一化 处理一维数据
def demo03():
    # 1. 创建样本数据
    # 1张图片,2个通道,3行4列(像素点)
    input_1d = torch.randn(size=(2, 2))
    print(f"input_2d:{input_1d}")

    # 2. 创建线性层
    linear1 = torch.nn.Linear(2, 4)

    # 3. 对数据进行 线性变换
    l1 = linear1(input_1d)
    print(f"l1:{l1}")

    # 4. 创建批量归一化层
    bn1d = torch.nn.BatchNorm1d(num_features=4)

    # 5. 对线性层处理结果l1 进行 批量归一化处理
    output_1d = bn1d(l1)
    print(f"output_1d:{output_1d}")


if __name__ == "__main__":
    demo01()
    demo02()
    demo03()
