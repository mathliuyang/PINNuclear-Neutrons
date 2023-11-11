'''
基于PINN深度机器学习技术求解 多维中子学扩散方程
临界条件下稳态扩散方程的验证
当系统处于稳态时, 形式为:
$$
\nabla^2 \phi(r)+\frac{k_{\infty} / k_{\text {eff }}-1}{L^2} \phi(r)=0
$$

当系统处于临界状态 $\left(k_{\mathrm{eff}}=1\right)$ 时, 为:
$$
\nabla^2 \phi(r)+B_g^2 \phi(r)=0
$$

式中, $B_g^2$ 为系统临界时的几何曲率, 与系统的几何特性相关，临界时等于材料曲率。

为了验证计算结果, 选取针对特定几何有解析解的扩散方程进行数值验证, 相关结论也可供其他形式的方程与几何形式参考。
验证计算神经网络架构均采用全连接方式, 激活函数选取具有高阶导数连续特点的双曲正切函数 $\mathrm{tanh}$,
其形式为 $\tanh (x)=\left(\mathrm{e}^x-\mathrm{e}^{-x}\right) /\left(\mathrm{e}^x+\mathrm{e}^{-x}\right)$,
网络初始值权重 $\{\vec{w}, \vec{b}\}$采用高斯分布随机采样 。

平板的解析解为: $C \cdot \cos (x \cdot \pi / a)$;球的解析解为: $C / r \cdot \sin (\pi \cdot r / R)$;

验证计算神经网络的超参数设定为: 深度 $l=16$, 中间层隐藏神经单元数量 $s=20$, 边界权重 $P_{\mathrm{b}}=100, C=0.5$,
几何网格点随机均布, 学习率从 0.001 开始, 训练至损失函数值 $f_{\text {Loss }}$ 在 100 次学习内不再下降结束.
'''

import deepxde as dde
import numpy as np
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


# 定义边界条件
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 定义数据
data = dde.data.PDE(geom, pde, bc, num_domain=898, num_boundary=2, solution=phi_analytical, num_test=100)
# 定义神经网络
layer_size = [1] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, Pb])
# 训练模型
losshistory, train_state = model.train(epochs=3500)
# 保存和可视化训练结果
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# 输出在 x=0 处的值(即 C)
print("Predicted value at x=0:", model.predict(np.array([0])))

# ---------------------------------------------------------------------------------------------
# 可视化
# 创建保存图像的文件夹
output_folder = "figure/算例1"
os.makedirs(output_folder, exist_ok=True)

# 设置中文字体

# 替换为您的字体文件路径
font_path = '/System/Library/Fonts/STHeiti Light.ttc'

# 添加字体路径
font_properties = FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_properties.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 绘制损失函数变化图
loss_train = np.sum(losshistory.loss_train, axis=1)
loss_test = np.sum(losshistory.loss_test, axis=1)

plt.figure(figsize=(10, 6))  # 设置图像大小
plt.semilogy(losshistory.steps, loss_train, label="训练损失")
plt.semilogy(losshistory.steps, loss_test, label="测试损失")
for i in range(len(losshistory.metrics_test[0])):
    plt.semilogy(
        losshistory.steps,
        np.array(losshistory.metrics_test)[:, i],
        label="测试指标",
    )
plt.xlabel("# 步骤")
plt.legend()
# 保存为png
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "losshistory.png"), format="png", dpi=200)


# 绘制训练状态变化图
def _pack_data(train_state):
    def merge_values(values):
        if values is None:
            return None
        return np.hstack(values) if isinstance(values, (list, tuple)) else values

    y_train = merge_values(train_state.y_train)
    y_test = merge_values(train_state.y_test)
    best_y = merge_values(train_state.best_y)
    best_ystd = merge_values(train_state.best_ystd)
    return y_train, y_test, best_y, best_ystd


if isinstance(train_state.X_train, (list, tuple)):
    print(
        "错误：网络有多个输入，尚未实现绘制此类结果。"
    )

y_train, y_test, best_y, best_ystd = _pack_data(train_state)
y_dim = best_y.shape[1]

# 回归分析图
if train_state.X_test.shape[1] == 1:
    idx = np.argsort(train_state.X_test[:, 0])
    X = train_state.X_test[idx, 0]
    plt.figure(figsize=(12, 5))  # 设置图像大小
    for i in range(y_dim):
        if y_train is not None:
            plt.plot(train_state.X_train[:, 0], y_train[:, i], "ok", label="训练点分布")
        if y_test is not None:
            plt.plot(X, y_test[idx, i], "-k", label="真实")
        plt.plot(X, best_y[idx, i], "--r", label="预测")
        if best_ystd is not None:
            plt.plot(
                X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% 置信区间"
            )
            plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "solution_state.png"), format="png", dpi=200)
# 神经网络输出的残差
if y_test is not None:
    plt.figure(figsize=(6, 5))  # 设置图像大小
    residual = y_test[:, 0] - best_y[:, 0]
    plt.plot(best_y[:, 0], residual, "o", zorder=1)
    plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
    plt.xlabel("预测值")
    plt.ylabel("残差 = 观测 - 预测")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "y_test.png"), format="png", dpi=200)
# 神经网络输出的标准差
if best_ystd is not None:
    plt.figure(figsize=(6, 5))  # 设置图像大小
    for i in range(y_dim):
        plt.plot(train_state.X_test[:, 0], best_ystd[:, i], "-b")
        plt.plot(
            train_state.X_train[:, 0],
            np.interp(
                train_state.X_train[:, 0], train_state.X_test[:, 0], best_ystd[:, i]
            ),
            "ok",
        )
    plt.xlabel("x")
    plt.ylabel("std(y)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "best_ystd.png"), format="png", dpi=200)

# 绘制数值解与解析解的图像并比较误差
x = np.linspace(-a / 2, a / 2, 100).reshape((-1, 1))
y_pred = model.predict(x)
y_exact = phi_analytical(x)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y_pred, label="神经网络解", color="tab:blue")
plt.plot(x, y_exact, label="解析解", color="tab:orange")
plt.title("数值解与解析解的比较")
plt.xlabel("x")
plt.ylabel("$\phi(x)$")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(y_pred - y_exact), label="绝对误差", color="tab:red")
plt.title("数值解与解析解的绝对误差")
plt.xlabel("x")
plt.ylabel("绝对误差")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "solution_comparison.png"), format="png", dpi=200)

plt.show()
