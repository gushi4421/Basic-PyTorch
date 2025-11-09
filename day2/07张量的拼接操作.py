"""
案例:
    演示张量的拼接操作

涉及API:
    cat():不改变维度数,拼接张量,除了拼接的那个维度外,其他维度数必须保持一致
    stack():会改变维度数,拼接张量,所有的维度数都必须保持一致
"""

import torch

# 1. 创建两个张量
t1 = torch.randint(1, 10, size=(2, 3))
print(f"t1:{t1}")

t2 = torch.randint(1, 10, (2, 3))
print(f"t2:{t2}")

# 2. 演示张量的拼接
# 思路1:cat()拼接张量
t3 = torch.cat([t1, t2], dim=0)  # (2,3) + (2,3) = (4,3)
print(f't3"{t3},shape:{t3.shape}')

t4 = torch.cat([t1, t2], dim=1)  # (2,3) + (2,3) = (2,6)
print(f"t4:{t4},shape:{t4.shape}")

# t5 = torch.cat([t1, t2], dim=2)  # 报错
# print(f"t5:{t5},shape:{t5.shape}")

print("-" * 30)
# 思路2:stack)拼接张量,可以是新维度,但是无论新旧维度,所有维度都必须保持一致
t6 = torch.stack([t1, t2], dim=0)  # (2,3)+(2,3)=(2,2,3)
print(f"t6:{t6},shape:{t6.shape}")
t7 = torch.stack([t1, t2], dim=1)  # (2,3)+(2,3)=(2,2,3)
print(f"t7:{t7},shape:{t7.shape}")
t8 = torch.stack([t1, t2], dim=2)  # (2,3)+(2,3)=(2,3,2)
print(f"t8:{t8},shape:{t8.shape}")
# t9 = torch.stack([t1, t2], dim=3)  # (2,3)+(2,3)=(2,3,?,2)  报错
# print(f"t8:{t9},shape:{t9.shape}")
