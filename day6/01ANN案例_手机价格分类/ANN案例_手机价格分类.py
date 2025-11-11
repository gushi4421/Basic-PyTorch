"""
案例:
    ANN(人工神经网络)案例: 手机价格分类案例

背景:
    基于手机的20列特征 -> 预测手机的价格区间

ANN案例的视线步骤:
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试

优化方法:
    1. SGD -> Adam
    2. lr: 0.001 -> 0.0001
    3. 增加网络的深度
    4. 增加训练的轮数: 50 -> 100
    5. 对数据进行标准化
"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary
from sklearn.preprocessing import StandardScaler


# 1. 定义函数,构建数据集
def create_dataset():
    # 1. 加载csv文件数据集
    data = pd.read_csv("day6/01ANN案例_手机价格分类/data/phone_price.csv")

    # 2. 获取x特征列 和 y标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # 3. 把特征列转为浮点型
    x = x.astype(np.float32)

    # 4. 切分训练集和测试集
    # 参1: 特征, 参2: 标签, 参3: 测试集所占比例, 参4: 随机种子, 参5: 样本的分布(即: 参考y的类别进行抽取数据)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=3, stratify=y
    )

    # 5. 把数据集封装成 张量数据集  思路: 数据 -> 张量 -> 数据集Dataset -> 数据加载器
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.tensor(y_train.values)
    )
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test.values))

    # 6. 返回结果                         20(充当输入特征数)   4(充当 输出标签)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# 2. 搭建神经网络
class PhonePriceModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        # 1.1 初始化父类成员
        super().__init__()

        # 1.2 搭建神经网络
        # 隐藏层1
        self.linear1 = torch.nn.Linear(input_dim, 64)
        # 隐藏层2
        self.linear2 = torch.nn.Linear(64, 32)
        # 隐藏层3
        self.linear3 = torch.nn.Linear(32, 16)
        # 输出层
        self.output = torch.nn.Linear(16, output_dim)

    def forward(self, x):
        # 2.1 隐藏层1: 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear1(x))
        # 2.2 隐藏层2: 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))
        # 2.3 隐藏层3: 加权求和 + 激活函数
        # 正常写法, 但是不需要, 后续用CrossEntropyLoss替代】【
        # x = torch.softmax(self.ouput(x), dim=1)
        x = torch.relu(self.linear3(x))
        # 2.4 输出层
        x = self.output(x)
        return x


# 3. 模型训练
def train(train_dataset, input_dim, output_dim):
    # 1. 创建数据加载器,流程: 数据 -> 张量 -> 数据集 -> 数据加载器
    # 参1: 数据集对象 参2: 每批次的数据条数 参3: 是否打乱数据
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 2. 创建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)
    # 3. 定义损失函数,
    criterion = torch.nn.CrossEntropyLoss()
    # 4. 创建优化器对象
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 5. 模型训练
    # 5.1 定义变量,记录训练的 总轮数
    epochs = 100
    # 5.2 开始(每轮)训练
    for epoch in range(epochs):
        # 5.2.1 定义变量,记录每次训练的损失值,训练批次数
        total_loss, batch_num = 0.0, 0
        # 5.2.2 定义变量,表示训练开始的时间
        start = time.time()
        # 5.2.3 开始本轮的 各个批次的训练
        for x, y in train_loader:
            # 5.2.4 切换模型状态
            model.train()  # 训练模式       mode.eval() #测试模式
            # 5.2.5 模型预测
            y_pred = model(x)
            # 5.2.6 计算损失
            loss = criterion(y_pred, y)
            # 5.2.7 反向传播,优化参数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()
            # 5.2.8 累加损失值
            total_loss += loss.item()  # 把本轮的每批次(16条)的 平均损失累计起来
            batch_num += 1

        # 5.2.4  本轮训练结束,打印训练信息
        print(f"epoch:{epoch+1},loss:{total_loss}")

    # 6. 走到这里,说明本轮训练结束,保存模型参数
    print(f"\n\n模型的参数信息:{model.state_dict()}\n\n")
    torch.save(
        model.state_dict(), "day6/01ANN案例_手机价格分类/model/model.pth"
    )  # 后缀名用:pth,pkl,pickle


# 4. 模型测试
def evaluate(test_dataset, input_dim, output_dim):
    # 1. 创建神经网络对象
    model = PhonePriceModel(input_dim, output_dim)
    # 2. 加载模型参数
    model.load_state_dict(torch.load("day6/01ANN案例_手机价格分类/model/model.pth"))
    # 3. 创建测试集的 数据加载器对象
    # 参1: 数据集对象 , 参2: 每批次的数据条数 , 参3: 是否打乱
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # 4. 定义变量, 记录预测正确的样本个数
    correct = 0
    # 5. 从数据加载器中,获取到每批次的数据day6/01ANN案例_手机价格分类/ANN案例_手机价格分类.py
    for x, y in test_loader:
        # 5.1 切换模型状态 -> 测试数据
        model.eval()
        # 5.2 测试数据
        y_pred = model(x)
        # 5.3 根据加权求和，获取预测结果，用argmax获取下标，就是预测结果
        y_pred = torch.argmax(y_pred, dim=1)  # dim=1表示逐行进行处理
        # 5.4 统计正确的样本数
        correct += (y_pred == y).sum()

    # 6. 计算准确率
    print(f"测试集上的准确率:{correct.item()/len(test_dataset):.4f}")


# 5. 测试
if __name__ == "__main__":
    # 1. 准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f"训练集 数据集对象:{train_dataset}")
    print(f"测试集 数据集对象:{test_dataset}")
    print(f"输入特征数:{input_dim}")
    print(f"输出特征数:{output_dim}")

    # 2. 搭建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)
    # 计算模型参数
    # 参1: 模型对象, 参2: 输入数据的形状(批次大小,输入特征数)
    # summary(model, input_size=(16, input_dim))

    # 4. 模型训练
    # train(train_dataset, input_dim, output_dim)

    # 5. 模型测试
    evaluate(test_dataset, input_dim, output_dim)
