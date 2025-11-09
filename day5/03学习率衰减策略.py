"""
案例:
    演示学习率衰减策略

学习率衰减策略介绍:
    目的:
        较之于AdaGrad,RMSProp,Adam方式,我们可以通过 等间隔,指定间隔,指数等方式,来手动控制学习率的调整

    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减

等间隔学习率衰减:
    step_suze: 间隔的轮数,即: 多少轮调整一次学习率
    gamma: 学习率衰减系数,即: lr新 = lr旧 * gamma

指定间隔学习率衰减:
    milestones: 需要调整学习率的指定轮数
    gamma: 学习率衰减系数,即: lr新 = lr旧 * gamma

指数间隔学习率衰减:
    前期学习率衰减快, 后面慢,更符合梯度下降的规律
    公式:
        lr新 = lr旧 * gamma ** epoch

总结:
    等间隔学习率衰减:
        优点:
            直观,易于调试,适用于 大批量数据
        缺点:
            学习率变化较大,可能跳过最优解
        应用场景:
            大型数据集,较为简单的任务

    指定学习率衰减:
        优点:
            易于调试,稳定训练过程
        缺点:
            在某些情况下可能衰减过快,导致优化提前停滞
        应用场景:
            对训练平稳性要求较高的任务

    指数学习率衰减:
        优点:
            平滑,且考虑历史更新,收敛稳定性较强
        缺点:
            超参调节较为复杂,可能需要更多资源
        应用场景:
            高精度训练,避免过快收敛

其实应用Adam更多
"""

import torch
import matplotlib.pyplot as plt


# 1. 定义函数,演示: 等间隔学习率衰减
def demo01():
    # 1. 定义变量,记录初始的 学习率  训练的轮数  每轮训练的批次数
    lr = 0.1
    epochs = 200
    iteration = 10

    # 2. 创建数据集,y_true,x,w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w,需要自动微分
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象,动量法 -> 加速模型的收敛,减少震荡
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数
    optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建等间隔学习率衰减对象
    # 参1: 优化器对象， 参2: 间隔的轮数(多少轮调整一次学习率), 参3: 学习率衰减系数
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 5.创建两个列表,分别表示: 训练轮数,每轮训练用的学习率
    epoch_list = []
    lr_list = []

    # 6. 循环遍历训练轮数,进行具体的训练
    for epoch in range(epochs):
        # 7. 获取当前轮数 和 学习率, 并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        # 8. 循环遍历,每轮每批次进行训练
        for batch in range(iteration):
            # 9. 先计算预测值,然后基于损失函数计算损失
            y_pred = w * x
            # 10. 计算损失,最小二乘法
            loss = (y_pred - y_true) ** 2
            # 11. 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 12. 更新学习率
        scheduler.step()

    # 13. 打印结果
    print(f"lr_list:{lr_list}")

    # 14. 可视化
    # x轴:训练的轮数,y轴: 每轮训练的学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# 2. 定义函数,演示: 指定间隔学习率衰减
def demo02():
    # 1. 定义变量,记录初始的 学习率  训练的轮数  每轮训练的批次数
    lr = 0.1
    epochs = 200
    iteration = 10

    # 2. 创建数据集,y_true,x,w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w,需要自动微分
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象,动量法 -> 加速模型的收敛,减少震荡
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数
    optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建等间隔学习率衰减对象
    # 参1: 优化器对象， 参2: 需要调整的轮数(在多少轮调整学习率), 参3: 学习率衰减系数
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.5
    )

    # 5.创建两个列表,分别表示: 训练轮数,每轮训练用的学习率
    epoch_list = []
    lr_list = []

    # 6. 循环遍历训练轮数,进行具体的训练
    for epoch in range(epochs):
        # 7. 获取当前轮数 和 学习率, 并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        # 8. 循环遍历,每轮每批次进行训练
        for batch in range(iteration):
            # 9. 先计算预测值,然后基于损失函数计算损失
            y_pred = w * x
            # 10. 计算损失,最小二乘法
            loss = (y_pred - y_true) ** 2
            # 11. 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 12. 更新学习率
        scheduler.step()

    # 13. 打印结果
    print(f"lr_list:{lr_list}")

    # 14. 可视化
    # x轴:训练的轮数,y轴: 每轮训练的学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# 3. 定义函数,演示: 指数学习率衰减
def demo03():
    # 1. 定义变量,记录初始的 学习率  训练的轮数  每轮训练的批次数
    lr = 0.1
    epochs = 200
    iteration = 10

    # 2. 创建数据集,y_true,x,w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w,需要自动微分
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3. 创建优化器对象,动量法 -> 加速模型的收敛,减少震荡
    # 参1: 待优化的参数, 参2: 学习率, 参3: 动量系数
    optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建等指数学习率衰减对象
    # 参1: 优化器对象， 参2: 间隔的轮数(多少轮调整一次学习率), 参3: 学习率衰减系数
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 5.创建两个列表,分别表示: 训练轮数,每轮训练用的学习率
    epoch_list = []
    lr_list = []

    # 6. 循环遍历训练轮数,进行具体的训练
    for epoch in range(epochs):
        # 7. 获取当前轮数 和 学习率, 并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        # 8. 循环遍历,每轮每批次进行训练
        for batch in range(iteration):
            # 9. 先计算预测值,然后基于损失函数计算损失
            y_pred = w * x
            # 10. 计算损失,最小二乘法
            loss = (y_pred - y_true) ** 2
            # 11. 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 12. 更新学习率
        scheduler.step()

    # 13. 打印结果
    print(f"lr_list:{lr_list}")

    # 14. 可视化
    # x轴:训练的轮数,y轴: 每轮训练的学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# 4. 测试
if __name__ == "__main__":
    demo01()
    demo02()
    demo03()
