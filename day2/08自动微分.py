"""
案例:
    演示自动微分模块,循环实现 计算梯度,更新参数

需求:
    求y=x^2+20的极小值点,并打印y是最小值时,w的值

解题步骤:
    1. 定义w=10 requires_grad=True  dtlosspe=torch.float32
    2. 定义 loss=w**2+2-
    3. 利用梯度下降算法 循环100次 求最优解
    3.1 正向计算(正向传播)
    3.2 梯度清零 w.grad.zero_()
    3.3 反向传播
    3.4 梯度更新 w.data = w.data - 0.01 * w.grad
"""

import torch

# 1. 定义点 w=10 requires_grad=True dtlosspe=torch.float32
# 参1: 初始值,参2:自动微分(求导),参3:数据类型,浮点型
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 2. 定义函数 loss=w**2 + 20
loss = w**2 + 20  # loss'=2w

# 3. 利用梯度下降法 循环迭代100次 求最优解
print(f"开始 权重初始值:{w},{0.01*w.grad}:无,loss:{loss}")

for i in range(100):
    # 3.1 正向计算(正向传播)
    loss = w**2 + 20
    
    # 3.2 梯度清零
    if w is not None:
        w.grad.zero_()
    
    # 3.3 反向传播
    loss.backward()
    
    # 3.4 梯度更新
    print(f"梯度值:{w.grad}")
    w.data = w.data - 0.01 * w.grad

    # 3.5 打印本次 梯度更新后的 权重参数结果
    print(f"第{i}次,权重初始值:{w},(0.01*w.grad:{0.01*w.grad:.5f},loss:{loss:.5f}")

# 4. 打印最终结果
print(f"权重:{w},梯度:{w.grad},loss:{loss}")
