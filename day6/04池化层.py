"""


特点:
    不会改变

"""

import torch
import matplotlib.pyplot as plt


# 1. 定义函数，演示单通道池化
def demo01():
    # 1. 创建1个 1通道 3*3的二维矩阵
    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=torch.float32)

    # 2. 创建最大池化层
    # 参1：池化核大小，参2：步长，参3：填充
    pool1 = torch.nn.MaxPool2d(2, 1, 0)
    outputs = pool1(inputs)
    print(f"outputs:{outputs},shape:{outputs.shape}")

    # 3. 创建平均池化层
    pool2 = torch.nn.AvgPool2d(2, 1, 0)
    outputs2 = pool2(inputs)
    print(f"outputs2:{outputs2},shape:{outputs2.shape}")


# 2. 定义函数，演示多通道池化
def demo02():
    # 1. 创建1个 3通道 3*3的二维矩阵
    inputs = torch.tensor(
        [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
        ],
        dtype=torch.float32,
    )

    # 2. 创建最大池化层
    # 参1：池化核大小，参2：步长，参3：填充
    pool1 = torch.nn.MaxPool2d(2, 1, 0)
    outputs = pool1(inputs)
    print(f"outputs:{outputs},shape:{outputs.shape}")

    # 3. 创建平均池化层
    pool2 = torch.nn.AvgPool2d(2, 1, 0)
    outputs2 = pool2(inputs)
    print(f"outputs2:{outputs2},shape:{outputs2.shape}")


# 3. 测试
if __name__ == "__main__":
    demo01()
    demo02()
