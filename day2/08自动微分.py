"""
案例:
    演示自动微分模块,循环实现 计算梯度,更新参数

需求:
    求y=x^2+20的极小值点,并打印y是最小值时,w的值

解题步骤:
    1. 定义w=10 requires_grad=True  dtype=torch.float32
    2. 定义 loss=w**2+20
    3. 利用梯度下降算法 循环100次 求最优解
    3.1 梯度清零 w.grad.zero_()
    3.2 正向计算(正向传播)
    3.3 反向传播
    3.4 梯度更新 w.data = w.data - 0.01 * w.grad
"""

import torch

# 1. 定义点 w=10 requires_grad=True dtype=torch.float32
w = torch.tensor(10.0, requires_grad=True)  # 简化写法

print(f"开始迭代,初始权重:{w.item():.5f}")

# 2. 利用梯度下降法 循环迭代100次 求最优解
for i in range(100):
    # 2.1 梯度清零
    if w.grad is not None:
        w.grad.zero_()

    # 2.2 正向计算
    loss = w**2 + 20

    # 2.3 反向传播
    loss.backward()

    # 2.4 梯度更新
    with torch.no_grad():  # 在更新时不计算梯度
        w.data -= 0.01 * w.grad

    # 2.5 打印进度
    if i % 10 == 0:  # 每10次打印一次
        print(
            f"第{i:2d}次, 权重:{w.item():.5f}, 梯度:{w.grad.item():.5f}, loss:{loss.item():.5f}"
        )

# 3. 打印最终结果
print(f"\n最终结果:")
print(f"权重:{w.item():.5f}, 梯度:{w.grad.item():.5f}, loss:{loss.item():.5f}")
print(f"理论最小值在w=0处,loss=20")
