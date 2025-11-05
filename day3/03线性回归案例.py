"""
案例:

一般步骤:
    1. 准备训练集数据
    2. 构建要使用的模型
    3. 设置损失函数与优化器
    4. 模型训练

涉及到的API:
    TensorDataset : 数据集对象
    DataLoader : 加载数据集
    torch.nn.Linear : 线性模型
    torch.nn.MSELoss : MSE损失函数
    torch.optim.SGD : SGD优化器

安装 matplotlib : pip install -U scikit-learn
numpy对象->张量Tensor->数据集对象TensorDataset->数据加载器DataLoader
"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 1. 准备训练集数据
def dataset():
    x, y, coef = make_regression(
        n_samples=100,  # 100个样本点
        n_features=1,  # 1个特征点
        noise=10,  # 噪声,噪声越大,样本点越散
        coef=True,  # 是否返回系数,默认为False,返回值为None
        bias=14.5,  # 偏置
        random_state=1,  # 随机种子,随机种子相同，输出数据相同
    )
    print(f"type:{type(x)}")  # numpy.ndarray

    # 2. 把上述的数据,封装成张量对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 3. 返回结果
    return x, y, coef


# 2 定义函数,表示模型训练
def train(x, y, coef):
    # 1. 创建数据集对象,把tensor -> 数据集对象 ->数据加载器
    dataset = TensorDataset(x, y)

    # 2. 创建数据加载器对象
    # 参1:数据集对象,参2:批次大小,参3:是否打乱数据(训练集打乱,测试机不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 创建初始的线性回归模型
    model = torch.nn.Linear(1, 1)

    # 4. 创建损失函数对象
    criterion = torch.nn.MSELoss()
    # 5. 创建优化器对象
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 6. 具体的训练过程
    # 6.1 定义变量,分别表示:训练轮数,每轮的(平均)损失值,训练总损失值,训练的样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0

    # 6.2 开始训练,按轮训练
    for epoch in range(epochs):
        # 6.3 每轮是分 批次 训练的,所以从 数据加载器中 获取 批次数据
        for train_x, train_y in dataloader:  # 7批次(16,16,16,16,16,16,4)
            # 6.3.1 模型预测
            y_pred = model(train_x)

            # 6.3.2 计算(每轮的平均)损失值
            loss = criterion(y_pred, train_y.reshape(-1, 1))  # -1 自动计算

            # 6.3.3 计算总损失 和 样本批次数
            total_loss += loss.item()
            total_sample += 1

            # 6.3.4 梯度清零,反向传播,梯度更新
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播,计算梯度
            optimizer.step()  # 梯度更新

        # 6.4 把本轮的(平均)损失值,添加到列表中
        loss_list.append(total_loss / total_sample)
        print(f"轮数:{epoch+1},平均损失:{total_loss/total_sample}")

    # 7. 打印最终的训练结果
    print(f"{epoch+1}轮的平均损失为:{total_loss/total_sample}")
    print(f"模型参数,权重:{model.weight},偏置：{model.bias}")

    # 8. 绘制损失函数
    plt.plot(range(epochs), loss_list)
    plt.title("损失值曲线变化图")
    plt.grid()  # 绘制网格线
    plt.show()

    # 9. 绘制预测值和真实值的关系
    # 9.1 绘制样本点分布情况
    plt.scatter(x, y)
    plt.show()

    # 9.2 绘制训练模型的预测值
    # x: 100哥样本点的特征
    y_pred = torch.tensor(data=[v * model.weight + model.bias for v in x])

    # 9.3 计算真实值
    y_true = torch.tensor(data=[v * coef + 14.5 for v in x])

    # 9.4 绘制 预测值 和 真实值的 折线图
    plt.plot(x, y_pred, color="red", label="预测值")
    plt.plot(x, y_true, color="green", label="真实值")

    # 9.5 图例,网格
    plt.legend()
    plt.grid()

    # 9.6 显示图像
    plt.show()


# 3. 测试
if __name__ == "__main__":
    # 3.1 创建数据集
    x, y, coef = dataset()
    # print(f"x:{x},y:{y},coef:{coef}")

    # 3.2 模型训练
    train(x, y, coef)
