"""
| 算例 | 网格点分布 |
| :---: | :---: |
| 1 | $D=D_0, \operatorname{card}\left(D_0\right)=2000$ |
| 2 | $D=D_0 \cup D_{\mathrm{Cl}}, \operatorname{card}\left(D_0\right)=1500$, <br> $\operatorname{card}\left(D_{\mathrm{cl}}\right)=500$ |
| 3 | $D=D_0 \cup D_{\mathrm{C} 2}, \operatorname{card}\left(D_0\right)=1500$ <br> $\operatorname{card}\left(D_{\mathrm{C} 2}\right)=500$ |
| 4 | $D=D_0 \cup D_{\mathrm{C} 2} \cup D_{\mathrm{B} 1} \cup D_{\mathrm{B} 2}$ <br> $\operatorname{card}\left(D_0\right)=1000, \operatorname{card}\left(D_{\mathrm{C} 2}\right)=500$, <br> $\operatorname{card}\left(D_{\mathrm{B} 1}\right)=250, \operatorname{card}\left(D_{\mathrm{B} 2}\right)=250$ |
各区域内网格点随机分布，其中
$$
\begin{aligned}
& D_0:\{(x, t) \mid-0.5 \leqslant x \leqslant 0.5,0 \leqslant t \leqslant 0.01\} \\
& D_{\mathrm{C} 1}:\{(x, t) \mid-0.5 \leqslant x \leqslant 0.5,0 \leqslant t \leqslant 0.002\} \\
& D_{\mathrm{C} 2}:\{(x, t) \mid-0.5 \leqslant x \leqslant 0.5,0 \leqslant t \leqslant 0.005\} \\
& D_{\mathrm{B} 1}:\{(x, t) \mid-0.5 \leqslant x \leqslant-0.45,0 \leqslant t \leqslant 0.01\} \\
& D_{\mathrm{B} 2}:\{(x, t) \mid 0.45 \leqslant x \leqslant 0.5,0 \leqslant t \leqslant 0.01\}
\end{aligned}
$$
"""
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# 定义各个区域的边界和点的数量
# 算例 1
# num_points_D0 = 2000
# num_points_DC1 = 0
# num_points_DC2 = 0
# num_points_DB1 = 0
# num_points_DB2 = 0
# 算例 2
# num_points_D0 = 1500
# num_points_DC1 = 500
# num_points_DC2 = 0
# num_points_DB1 = 0
# num_points_DB2 = 0
# 算例 3
# num_points_D0 = 1500
# num_points_DC1 = 0
# num_points_DC2 = 500
# num_points_DB1 = 0
# num_points_DB2 = 0
# 算例 4
num_points_D0 = 1000
num_points_DC1 = 0
num_points_DC2 = 500
num_points_DB1 = 250
num_points_DB2 = 250

# 生成D0区域的随机点
x_D0 = np.random.uniform(-0.5, 0.5, num_points_D0)
t_D0 = np.random.uniform(0, 0.01, num_points_D0)

# 生成DC1区域的随机点
x_DC1 = np.random.uniform(-0.5, 0.5, num_points_DC1)
t_DC1 = np.random.uniform(0, 0.002, num_points_DC1)

# 生成DC2区域的随机点
x_DC2 = np.random.uniform(-0.5, 0.5, num_points_DC2)
t_DC2 = np.random.uniform(0, 0.005, num_points_DC2)

# 生成DB1区域的随机点
x_DB1 = np.random.uniform(-0.5, -0.45, num_points_DB1)
t_DB1 = np.random.uniform(0, 0.01, num_points_DB1)

# 生成DB2区域的随机点
x_DB2 = np.random.uniform(0.45, 0.5, num_points_DB2)
t_DB2 = np.random.uniform(0, 0.01, num_points_DB2)

# 合并各个区域的点
x = np.concatenate((x_DC1, x_DC2, x_DB1, x_DB2))
t = np.concatenate((t_DC1, t_DC2, t_DB1, t_DB2))

# 打印生成的点的数量
print(f"Total Points: {len(x)}")

# 绘制D0区域的点
plt.scatter(x_D0, t_D0, label='D0', s=8, alpha=0.5)

# 绘制DC1区域的点
plt.scatter(x_DC1, t_DC1, label='DC1', s=8, alpha=0.5)

# 绘制DC2区域的点
plt.scatter(x_DC2, t_DC2, label='DC2', s=8, alpha=0.5)

# 绘制DB1区域的点
plt.scatter(x_DB1, t_DB1, label='DB1', s=8, alpha=0.5)

# 绘制DB2区域的点
plt.scatter(x_DB2, t_DB2, label='DB2', s=8, alpha=0.5)

# 添加标签和标题
plt.xlabel('x')
plt.ylabel('t')
plt.title('Grid Point Distribution')
plt.legend()
plt.grid(True)

# 显示图像
plt.show()


# 初始化参数
k_eff = 1.00202  # 有效增殖系数
k_inf = 1.00410  # 无穷增殖系数
epsilon = 0.001  # 临界判断阈值
a = 1  # 平板的宽度
D = 0.211e-2  # 扩散系数
v = 2.2e3  # 中子速度
L2 = 2.1037e-4  # 系统临界时的扩散长度
B2 = (np.pi / a) ** 2  # 系统临界时的几何曲率
l = 5  # 神经网络的深度
s = 20  # 神经网络的中间层隐藏神经单元数量
Pi = 50  # 初值权重
Pb = 100  # 边界权重
Pc = 100  # 额外配置点权重
C = 0.5  # 解析解参数
k_n = 1.0
l_n = 1.0
num_terms = 10  # 考虑的项数


# 定义初值条件
def phi_0(x):
    return np.cos(np.pi * x[:, 0:1] / a) - 0.4 * np.cos(2 * np.pi * x[:, 0:1] / a) - 0.4

# 定义解析解
# def phi_analytical(x):
#         phi_x_t = 0
#         for n in range(1, num_terms + 1):
#             integral_term = 2 / a * np.trapz(phi_0(x) * np.cos((2 * n - 1) * np.pi / a * x[:, 0:1]), x[:, 0:1])
#             exponential_term = np.cos((2 * n - 1) * np.pi / a * x[:, 0:1]) * np.exp((k_n - 1) * x[:, 1:2] / l_n)
#             phi_x_t += integral_term * exponential_term
#         return phi_x_t

# 定义几何网格
geom = dde.geometry.Interval(-a / 2, a / 2)
timedomain = dde.geometry.TimeDomain(0, 0.015)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# 定义微分方程
def pde(x, phi):
    dphi_xx = dde.grad.hessian(phi, x, i=0, j=0)
    dphi_t = dde.grad.jacobian(phi, x, i=0, j=1)
    return 1 / (D * v) * dphi_t - dphi_xx - (k_inf / k_eff - 1) / L2 * phi


# 定义边界条件
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 定义初值条件
ic = dde.IC(geomtime, lambda x: phi_0(x), lambda _, on_initial: on_initial)
# 定义数据
data = dde.data.TimePDE(
    geomtime, pde, [ic, bc], num_domain=num_points_D0, num_boundary=200, num_initial=200, solution=None, anchors=np.vstack((x, t)).T)
# 定义神经网络
layer_size = [2] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, loss_weights=[1, Pi, Pb])
losshistory, train_state = model.train(epochs=1000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
