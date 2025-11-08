"""
案例:
    演示 回归任务的损失函数介绍

回归任务常用损失函数如下:
    MAE: Mean Absolute Error, 平均绝对误差
        公式:
            误差绝对值之和 / 样本总数
        类似于L1正则化,权重可以降维0,数据会变得稀疏

        弊端:
            在0点不平滑,可能错过最小值

    MSE: Mean Squared Error, 均方误差
        公式:
            误差平方之和 / 样本总数
         弊端
            如果差值多大,可能存在梯度爆炸的情况

    Smooth L1:
        基于MAE与MSE做的一个综合,在[-1,1]是L2(MSE),其它段时L1
        这样既解决了L1(MAE)不平滑的问题(0点不可导,可能错过最小值)
        又解决了L2(MSE)的 梯度爆炸问题
"""

import torch


# 1. 定义函数,演示:MAE 损失含糊
def demo01():
    # 1. 定义变量,记录:真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    # 2. 定义变量,记录:预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3. 创建MAE损失函数对象
    criterion = torch.nn.L1Loss()

    # 4. 计算损失
    loss = criterion(y_pred, y_true)
    print(f"MAE:{loss}")


# 2. 定义函数,演示:MSE 损失函数
def demo02():
    # 1. 定义变量,记录:真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    # 2. 定义变量,记录:预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3. 创建MSE损失函数对象
    criterion = torch.nn.MSELoss()

    # 4. 计算损失
    loss = criterion(y_pred, y_true)
    print(f"MSE:{loss}")


# 3. 定义函数,演示: Smooth L1 损失函数
def demo03():
    # 1. 定义变量,记录:真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    # 2. 定义变量,记录:预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3. 创建SmoothL1损失函数对象
    criterion = torch.nn.SmoothL1Loss()

    # 4. 计算损失
    loss = criterion(y_pred, y_true)
    print(f"Smooth L1:{loss}")


# 4. 测试
if __name__ == "__main__":
    demo01()  # 0.699999
    demo02()  # 0.670000
    demo03()  # 0.335000
