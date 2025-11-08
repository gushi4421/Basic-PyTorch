import torch
import torch.nn.functional as F

# 只验证 input 参数
x = torch.tensor([[1.0, 2.0, 3.0],
                  [0.0, 0.0, 0.0]])  # 第二行全 0
y = F.softmax(x, dim=1)  # 沿行做 softmax
print("输入张量:\n", x)
print("Softmax 结果:\n", y)
print("每行求和:", y.sum(dim=1))