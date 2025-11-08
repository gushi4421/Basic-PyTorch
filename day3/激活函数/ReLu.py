import torch
import matplotlib.pyplot as plt

# 函数
x = torch.linspace(-20, 20, 1000)
y = torch.relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 导函数
x = torch.linspace(-20, 20, 1000, requires_grad=True)
y = torch.relu(x)
y.sum().backward()  # 对y求和后反向传播以计算梯度（即导数）

plt.plot(x.detach(), x.grad)  # x.detach()用于分离张量，避免绘图时的梯度计算问题
plt.grid()
plt.show()
