"""
演示张量和Numpy之间如何相互转换,以及如何从标量张量中提取内容


设计到的API:
    场景1 : 张量->numpy nd数组对象
        张量对象.numpy()    共享内存
        张量对象.numpy().copy()     不共享内存，链式编程写法
    场景2 : numpy nd数组->张量
        from_numpy()        共享内存
        torch.tensor(nd数组)    不共享内存
    场景3 : 从标量张量中提取其内容
        标量张量.item()

学习目标：
    1. 张量->numpy:     张量对象.numpy()
    2. numpy->张量：    torch.tensor(nd数组)
    3. 从标量张量中提取其内容： 标量张量.item()
"""

import torch
import numpy as np


# 1. 将张量转换为Numpy数组
def demo01():
    # 1. 创建张量
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f"t1:{t1},type:{type(t1)}")

    # 2. 张量->numpy
    n1 = t1.numpy()
    print(f"n1:{n1},type:{type(n1)}")

    # 3. 演示上述方式 共享内存
    n1[0] = 100
    print(f"n1:{n1}")  # [100, 2, 3, 4, 5]
    print(f"t1:{t1}")  # [100, 2, 3, 4, 5]


# 2. 将Numpy数组转换为张量
def demo02():
    # 1. 创建numpy数组
    n1 = np.array([1, 2, 3])
    print(f"n1:{n1},type:{n1}")

    # 2. 将上述numpy数组转换成张量
    # t1=torch.from_numpy(n1).type(torch.float32)
    t1 = torch.from_numpy(n1)  # 共享内存
    print(f"t1:{t1},type:{type(t1)}")

    t2 = torch.tensor(n1)  # 不共享内存
    print(f"t2:{t2},type:{type(t2)}")

    # 3. 演示上述方式 共享内存
    n1[0] = 100
    print(f"n1:{n1}")  # 100 2 3
    print(f"t1:{t1}")  # 100 2 3
    print(f"t2:{t2}")  # 1 2 3


# 3. 标量张量和数字转换
def demo03():
    # 1. 创建张量
    # t1 = torch.tensor(100)    #true
    t1 = torch.tensor([100])  # true
    # t1 = torch.tensor([100,200])  #false
    # t1=torch.tensor('王') # false , 张量只能是一个 数值
    print(f"t1:{t1},type:{type(t1)}")

    # 2. 从张量中提取内容
    value = t1.item()
    print(f"value:{value},type:{type(value)}")


if __name__ == "__main__":
    demo01()
    demo02()
    demo03()
