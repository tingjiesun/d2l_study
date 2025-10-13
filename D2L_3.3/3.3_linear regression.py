import numpy as np                 # 导入 NumPy（本例中未直接使用，但常见依赖）
import torch                       # 导入 PyTorch
from torch.utils import data       # 导入 PyTorch 数据工具（TensorDataset、DataLoader）
from d2l import torch as d2l       # 导入 d2l 的 PyTorch 版辅助函数
from torch import nn               # 导入神经网络模块

true_w = torch.tensor([2, -3.4])   # 线性模型的真实权重（2 个特征）
true_b = 4.2                       # 线性模型的真实偏置
features, labels = d2l.synthetic_data(true_w, true_b, 1000)  # 生成 1000 条合成数据 (X, y)

def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)           # 将 (features, labels) 打包成数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 构造按批次迭代的数据加载器

batch_size = 10                                          # 每个小批量大小
data_iter = load_array((features, labels), batch_size)   #生成训练用的数据迭代器

next(iter(data_iter))                                    # 取出第一批（用于快速检查迭代器输出）

net = nn.Sequential(nn.Linear(2, 1))                     # 定义顺序模型：单层线性回归（2->1）

net[0].weight.data.normal_(0, 0.01)                      # 权重用 N(0, 0.01) 正态随机初始化
net[0].bias.data.fill_(0)                                # 偏置初始化为 0

loss = nn.MSELoss()
loss_1=nn.HuberLoss# 均方误差损失函数（回归常用）

trainer = torch.optim.SGD(net.parameters(), lr=0.03)     # 优化器：SGD，学习率 0.03

num_epochs = 3                                           # 训练总轮数
for epoch in range(num_epochs):                          # 外层循环：遍历每个 epoch
    for X, y in data_iter:                               # 内层循环：按小批量读取数据
        l = loss_1(net(X), y)                              # 前向计算并求当前批次损失
        trainer.zero_grad()                              # 清空上一次累积的梯度
        l.backward()                                     # 反向传播，计算参数梯度
        trainer.step()                                   # 参数更新（一步优化）
    l = loss(net(features), labels)                      # 用整集数据评估该轮损失
    print(f'epoch {epoch + 1}, loss {l:f}')              # 打印当前轮的损失

w = net[0].weight.data                                   # 取出学习到的权重
print('w的估计误差:', true_w - w.reshape(true_w.shape))  # 打印权重估计误差（与真实 w 的差）
b = net[0].bias.data                                     # 取出学习到的偏置
print('b的估计误差:', true_b - b)                         # 打印偏置估计误差（与真实 b 的差）
