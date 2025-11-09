"""
案例:
    用PyTorch演示线性回归模型
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

安装 matplotlib: pip install matplotlib
安装 scikit-learn: pip install scikit-learn
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
        coef=True,  # 是否返回系数
        bias=14.5,  # 偏置
        random_state=1,  # 随机种子
    )
    print(f"x type:{type(x)}, shape:{x.shape}")  # numpy.ndarray

    # 2. 把上述的数据,封装成张量对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # 重塑为列向量

    # 3. 返回结果
    return x, y, coef


# 2 定义函数,表示模型训练
def train(x, y, coef):
    # 1. 创建数据集对象
    dataset = TensorDataset(x, y)

    # 2. 创建数据加载器对象
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 创建初始的线性回归模型
    model = torch.nn.Linear(1, 1)  # 输入1维，输出1维

    # 4. 创建损失函数对象和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5. 训练过程
    epochs = 100
    loss_list = []

    print("开始训练...")
    for epoch in range(epochs):
        # 每轮开始时重置统计量
        total_loss, total_sample = 0.0, 0

        for train_x, train_y in dataloader:
            # 前向传播
            y_pred = model(train_x)

            # 计算损失
            loss = criterion(y_pred, train_y)
            total_loss += loss.item()
            total_sample += 1

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新

        # 记录本轮平均损失
        avg_loss = total_loss / total_sample
        loss_list.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"轮数:{epoch+1}, 平均损失:{avg_loss:.4f}")

    # 6. 打印最终训练结果
    print(f"\n训练完成!")
    print(f"最终平均损失:{avg_loss:.4f}")
    print(f"模型参数 - 权重:{model.weight.item():.4f}, 偏置:{model.bias.item():.4f}")
    print(f"真实参数 - 权重:{coef:.4f}, 偏置:14.5")

    # 7. 绘制损失函数曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), loss_list)
    plt.title("损失值曲线变化图")
    plt.xlabel("训练轮数")
    plt.ylabel("损失值")
    plt.grid(True)

    # 8. 绘制预测结果
    plt.subplot(1, 2, 2)
    # 绘制样本点
    plt.scatter(x.detach().numpy(), y.detach().numpy(), alpha=0.6, label="样本点")

    # 绘制预测线
    with torch.no_grad():
        y_pred = model(x)
        plt.plot(
            x.detach().numpy(),
            y_pred.detach().numpy(),
            color="red",
            linewidth=2,
            label="预测线",
        )

    # 绘制真实线（无噪声）
    x_range = torch.linspace(x.min(), x.max(), 100)
    y_true = x_range * coef + 14.5
    plt.plot(
        x_range.detach().numpy(),
        y_true.detach().numpy(),
        color="green",
        linestyle="--",
        label="真实线",
    )

    plt.title("线性回归拟合结果")
    plt.xlabel("特征值")
    plt.ylabel("目标值")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 3. 测试
if __name__ == "__main__":
    # 创建数据集
    x, y, coef = dataset()
    print(f"数据集形状: x{x.shape}, y{y.shape}")
    print(f"真实系数: 权重={coef:.4f}, 偏置=14.5")

    # 模型训练
    train(x, y, coef)
