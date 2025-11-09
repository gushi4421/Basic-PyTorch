"""
案例:
    演示参数初始化的7种方式

参数初始化的目的:
    1. 防止梯度消失 或者 梯度爆炸
    2. 提高收敛速度
    3. 打破对称性

参数初始化的方式:
    无法打破对称性的:
        全0,全1,固定值
    可以打破对称性的:
        随机初始化,正态分布初始化,kaiming初始化,xavier初始化


总结:
    1. 记忆kaiming初始化,xavier初始化,全0初始化
    2. 关于初始化的选择上:
        激活函数ReLu及其系列:优先使用 kaiming
        激活函数非ReLu:优先用xavier
        如果是浅层网络:可以考虑使用 随机初始化
"""

import torch


# 1. 均匀分布随机初始化
def demo01():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)
    # 2. 对权重(w)进行随机初始化,从0-1均匀分布产生参数
    torch.nn.init.uniform_(linear.weight)
    # 3. 对偏置(b)进行随机初始化,从0-1均匀分布产生参数
    torch.nn.init.uniform_(linear.bias)
    # 4. 打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)


# 2. 固定初始化
def demo02():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)
    # 2. 对权重(w)进行初始化,设置固定值为3
    torch.nn.init.constant_(linear.weight, 3)
    # 3. 对偏置(b)进行初始化,设置固定值为3
    torch.nn.init.constant_(linear.bias, 3)
    # 4. 打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)


# 3. 全0初始化
def demo03():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)
    # 2. 对权重(w)进行全0初始化
    torch.nn.init.zeros_(linear.weight)
    # 3. 对偏置(b)进行全0初始化
    torch.nn.init.zeros_(linear.bias)
    # 4. 打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)


# 4. 全1初始化
def demo04():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)
    # 2. 对权重(w)进行全1初始化
    torch.nn.init.ones_(linear.weight)
    # 3. 对偏置(b)进行全1初始化
    torch.nn.init.ones_(linear.bias)
    # 4. 打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)


# 5. 正态分布随机初始化
def demo05():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)
    # 2. 对权重(w)进行正态分布初始化
    torch.nn.init.normal_(linear.weight)
    # 3. 对偏置(b)进行正态分布初始化
    torch.nn.init.normal_(linear.bias)
    # 4. 打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)


# 6. kaiming初始化
def demo06():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)

    # 2. 对权重(w)进行kaiming初始化
    # 2.1 kaiming正态分布初始化
    # torch.nn.init.kaiming_normal_(linear.weight)

    # 2.2 kaiming均匀分布初始化
    torch.nn.init.kaiming_uniform_(linear.weight)

    # 3. 打印生成结果
    print(linear.weight.data)


# 7. xavier初始化
def demo07():
    # 1. 创建1个线性层,输入维度设置为5,输出维度设置为3
    linear = torch.nn.Linear(5, 3)

    # 2. 对权重(w)进行xavier初始化
    # 2.1 xavier正态分布初始化
    # torch.nn.init.xavier_normal_(linear.weight)

    # 2.2 xavier均匀分布初始化
    torch.nn.init.xavier_uniform_(linear.weight)

    # 3. 打印生成结果
    print(linear.weight.data)


if __name__ == "__main__":
    demo01()  # 均匀分布随机初始化
    demo02()  # 固定初始化
    demo03()  # 全0初始化
    demo04()  # 全1初始化
    demo05()  # 正态分布随机初始化
    demo06()  # kaiming初始化
    demo07()  # cavier初始化
