"""
案例:
    演示近30天,天气分布情况

结论:
    针对于β(调节权重系数)来讲,其值越大说明:越依赖指数加权平均,越不依赖本地的梯度值,数据就越平缓

指数移动加权平均(EMA)本质上是一种数学上的“算法思想”或“数据处理技术”，而不是一个完整的、独立的梯度下降优化算法。
"""

import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30


# 1. 实际平均温度
def test01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = (
        torch.randn(
            size=[
                ELEMENT_NUMBER,
            ]
        )
        * 10
    )
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color="r")
    plt.scatter(days, temperature)
    plt.show()


# 2. 指数加权平均温度
def test02(beta):
    torch.manual_seed(0)  # 固定随机数种子
    temperature = (
        torch.randn(
            size=[
                ELEMENT_NUMBER,
            ]
        )
        * 10
    )  # 产生30天的随机温度
    exp_weight_avg = []
    for idx, temp in enumerate(temperature, 1):  # 从下标1开始
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color="r")
    plt.scatter(days, temperature)
    plt.show()


if __name__ == "__main__":
    test01()
    test02(0.5)
    test02(0.9)
