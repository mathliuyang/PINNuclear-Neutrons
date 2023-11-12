
import deepxde as dde
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 初始化参数
k_eff = 1  # 有效增殖系数
a = 1  # 平板的宽度
B2 = (np.pi / a) ** 2  # 系统临界时的几何曲率
l = 16  # 神经网络的深度
s = 20  # 神经网络的中间层隐藏神经单元数量
Pb = 100  # 边界权重
Pc = 100 # 额外配置点权重
C = 0.5  # 解析解参数


# 定义解析解
def phi_analytical(x):
    return C * np.cos(x * np.pi / a)


# 定义几何网格
geom = dde.geometry.Interval(-a / 2, a / 2)


# 定义微分方程
def pde(x, phi):
    dphi_xx = dde.grad.hessian(phi, x, i=0, j=0)
    return dphi_xx + B2 * phi


# # 定义边界条件
# bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 扩散方程特征向量加速收敛方法验证
observe_x = np.array([0])
observe_phi0 = dde.icbc.PointSetBC(observe_x, np.array([0.5]))
# 定义数据
data = dde.data.PDE(geom, pde, [observe_phi0], num_domain=898, num_boundary=2, anchors=observe_x, solution=phi_analytical, num_test=900)
# 定义神经网络
layer_size = [1] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)

# 定义硬边界条件
def output_transform(x, y):
    return (x + a / 2) * (x - a / 2) * y
net.apply_output_transform(output_transform)
# ------------------------------------------------初始模型-------------------------------------------------
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, Pc])
# 修改网络的偏置项，调整初始C值
# for module in net.modules():
#     if isinstance(module, torch.nn.Linear):
#         module.bias.data = torch.tensor([1.], requires_grad=True)

# 输出初始网络在 x=0 处的值
print("Predicted value  of the initial network at x=0 : {:f}".format(model.predict(np.array([0]))[0]))

# 训练模型
model.train(epochs=1000)
# 输出在 x=0 处的值(即 C)
print("Predicted value at x=0: {:f}".format(model.predict(np.array([0]))[0]))
# 确保文件夹路径存在
output_folder = "model/算例2/model7"
os.makedirs(output_folder, exist_ok=True)

model_path = "model/算例2/model7.pth"
# 保存模型
torch.save(model.net.state_dict(), model_path)

# ------------------------------------------------迁移学习-------------------------------------------------
# 扩散方程特征向量加速收敛方法验证
observe_x = np.array([0])
observe_phi0 = dde.icbc.PointSetBC(observe_x, phi_analytical(observe_x))
# 定义数据
data = dde.data.PDE(geom, pde, [observe_phi0], num_domain=898, num_boundary=2, anchors=observe_x, solution=phi_analytical, num_test=900)
# 定义模型
loaded_model = dde.Model(data, net)
# 加载模型的状态字典
loaded_model.net.load_state_dict(torch.load(model_path))
print("迁移学习: Predicted value at x=0: {:f}".format(model.predict(np.array([0]))[0]))
# 训练模型
loaded_model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, Pc])
losshistory, train_state = loaded_model.train(epochs=1000)
print("迁移学习: Predicted value at x=0: {:f}".format(model.predict(np.array([0]))[0]))

# ------------------------------------------------可视化-------------------------------------------------
