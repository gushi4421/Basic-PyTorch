"""
案例:
    演示 梯度下降优化方法

梯度下降相关介绍:
    概述:
        梯度下降是结合 本次损失函数的导数(作为梯度)基于学习率 来更新权重的
    公式:
        W新 = W旧 - 学习率 * 梯度 0.01
    存在的问题:
        1. 遇到平缓区域,梯度下降(权重更新)可能会慢
        2. 可能会遇到 鞍点(梯度为0)
        3. 可能会遇到 局部最小值
    解决思路:
        从上述的 学习率 或者 梯度入手,进行优化,于是有了:动量法momentum,自适应学习率AdaGrad,RMSProp,综合衡量:Adam


    动量法Momentum:
        公式:
            St = β * St - 1 + (1 - β)*Gt
            解释:
                St: 本次的指数移动加权平均结果
                β: 调节权重系数,越大,数据越平缓,历史指数移动加权平均 比重越大,本次梯度权重越小
                St-1: 历史的指数移动加权平均结果
                Gt: 本次计算出的梯度(不考虑历史梯度)
        加入动量法后的 梯度更新公式:
            W新 = W旧 - 学习率 * St


    自适应学习率:AdaGrad
        公式:
            St=St-1 + Gt * Gt   累计平方梯度
            解释:
                St: 累计平方梯度
                St-1: 历史累计平方梯度
                Gt: 本次的梯度
        学习率η=η/(sqrt(St)+小常数)
            解释:
                小常数:1e-10,目的:防止分母变为0
        梯度下降公式:
            W新 = W旧 - 学习率 * Gt
        缺点:
            肯呢个会导致学习率过早,过量的降低,导致模型后期学习率太小,很难找到最优解


    AdaGrad优化算法: RMSProp -> 加入 调和权重系数
        公式:
            St=β*St-1 + (1 - β) * Gt * Gt   指数加权平均累计历史平方梯度
            解释:
                St: 累计平方梯度
                St-1: 历史累计平方梯度
                Gt: 本次的梯度
                β: 调和权重系数
        学习率η=η/(sqrt(St)+小常数)
            解释:
                小常数:1e-10,目的:防止分母变为0
        梯度下降公式:
            W新 = W旧 - 学习率 * Gt
        优点:
            RMSProp通过引入 衰减系数β,控制历史梯度 对 历史梯度信息获取的多少


    自适应据估计: Adam  -> RMSProp + Momentum
        思路:
            既优化学习率,又优化梯度
        公式:
            一阶矩: 算均值
                Mt = β1 * Mt-1 + (1 - β1) * Gt      充当: 梯度
                St = β2 * St-1 + (1 - β2) * Gt * Gt       充当: 梯度
            二阶矩: 梯度的方差
                Mt^ = Mt / (1 - β1 ^ t)
                St^ = St / (1 - β2 ^ t)
            权重更新公式:
                W新 = W旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^


总结: 如何选择梯度下降优化方法
    简单任务和较小的模型:
        SGD,动量法
    复杂任务或者有大量数据:
        Adam
    需要处理稀疏数据或者文本数据
        AdaGrad,RMSProp

"""

import torch


# 1. 演示动量法Momentum
def momentum():
    # 1。 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = (w**2) / 2.0

    # 3. 创建优化器(函数对象)->基于SGD(随机梯度下降),加入参数momentum,就是 动量法
    # 参1: (待优化的)参数列表,参2: 学习率,参3: 动量参数
    optimizer = torch.optim.SGD(
        params=[w], lr=0.01, momentum=0.9
    )  # 细节:momentum=0(默认),只考虑:本次梯度

    # 4. 计算梯度值:梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 5.重复上述的步骤,第2次更新权重参数
    criterion = (w**2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")


# 2. 演示AdaGrad
def adagrad():
    # 1。 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = (w**2) / 2.0

    # 3. 创建优化器(函数对象)->基于Adagrad(自适应学习率梯度下降)
    # 参1: (待优化的)参数列表,参2: 学习率
    optimizer = torch.optim.Adagrad(params=[w], lr=0.01)

    # 4. 计算梯度值:梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 5.重复上述的步骤,第2次更新权重参数
    criterion = (w**2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")


# 3. 演示RMSProp
def rmsprop():
    # 1。 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = (w**2) / 2.0

    # 3. 创建优化器(函数对象)->基于RMSProp(自适应学习率梯度下降)
    # 参1: (待优化的)参数列表,参2: 学习率,参3: 调和权重系数
    optimizer = torch.optim.RMSprop(params=[w], lr=0.01, alpha=0.9)  # alpha默认为0.99

    # 4. 计算梯度值:梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 5.重复上述的步骤,第2次更新权重参数
    criterion = (w**2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")


# 4. 演示Adam
def adam():
    # 1。 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = (w**2) / 2.0

    # 3. 创建优化器(函数对象)->基于Adam(自适应矩估计)
    # 参1: (待优化的)参数列表,参2: 学习率,参3: 调和权重系数
    optimizer = torch.optim.Adam(
        params=[w], lr=0.01, betas=(0.9, 0.999)
    )  # betas = (梯度用的 衰减系数 , 学习率用的 衰减系数)

    # 4. 计算梯度值:梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")

    # 5.重复上述的步骤,第2次更新权重参数
    criterion = (w**2) / 2.0
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")


# 5. 测试
if __name__ == "__main__":
    momentum()
    adagrad()
    rmsprop()
    adam()
