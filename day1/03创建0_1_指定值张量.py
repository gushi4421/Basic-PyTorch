"""
案例:
    创建值全部为0,1,指定值的张量

设计到的函数如下：
    torch.ones 和 torch.ones_like 创建全1的张量
    torch.zeros 和 torch.zeros_like 创建全0的张量
    torch.full 和 torch.full_like 创建全为指定值的张量

需要掌握的函数：
    zeros,full
    zeros较为常用
自动导包：
"""

import torch

# 场景：torch.ones 和 torch.ones_like 创建全1的张量
t1 = torch.ones(2,3)    #创建2行三列的全1张量
print(f't1:{t1},type:{type(t1)}')
print('-'*30)

# t2:3行2列
t2 = torch.tensor([1,2],[3,4],[5,6])
print(f't2:{t2},type:{type(t2)}')
print('-'*30)

# t3-> 基于t2的形状，创建全1张量
t3=torch.ones_like(t2)
print(f't3:{t3},type:{type(t3)}')
print('-'*30)

##################################################
# 场景：torch.zeros 和 torch.zeros_like 创建全0的张量
t1 = torch.zeros(2,3)    #创建2行三列的全0张量
print(f't1:{t1},type:{type(t1)}')
print('-'*30)

# t2:3行2列
t2 = torch.tensor([1,2],[3,4],[5,6])
print(f't2:{t2},type:{type(t2)}')
print('-'*30)

# t3-> 基于t2的形状，创建全0张量
t3=torch.zeros_like(t2)
print(f't3:{t3},type:{type(t3)}')
print('-'*30)

##################################################
# 场景：torch.full 和 torch.full_like 创建全为指定值的张量
t1 = torch.full(size=(2,3),fill_value=255)    #创建2行三列的全255张量
print(f't1:{t1},type:{type(t1)}')
print('-'*30)

# t2:3行2列
t2 = torch.tensor([1,2],[3,4],[5,6])
print(f't2:{t2},type:{type(t2)}')
print('-'*30)

# t3-> 基于t2的形状，创建全255张量
t3=torch.zeros_like(t2,fill_value=255)
print(f't3:{t3},type:{type(t3)}')
