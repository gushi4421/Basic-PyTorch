"""
演示张量的基本运算

涉及到的API:
    add(),sub(),mul(),div(),neg()   ->对应 加,减,乘,除,取反     add,substract,multiply,divide
    add_(),sub_(),mul_(),div_(),neg_()  ->功能同上,只不过可以修改源数据,类似于Pandas部分的inplace=True

需要你记忆的：
    1. 可以用符号替代  +,-,*./
    2. 如果是张量和数值运算,则:该数值会和张量中的每个值依次进行对应的运算
"""

import torch

# 1. 创建张量
t1 = torch.tensor([1, 2, 3])

# 2. 演示 加法
# t2 = t1.add(10)  # 不会修改源数据
# t2 = t1.add_(10)    #会修改源数据
t2 = t1 + 10  # 效果同t2 = t1.add(10)

# t1.add_(10)
# t1+=10    #效果同上

# 3. 打印结果
print(f"t1:{t1}")
print(f"t2:{t2}")


##### 其他函数效果同上，不多做演示 #####
