"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

output_folder = "figure/算例4/RAR"
os.makedirs(output_folder, exist_ok=True)

# 初始化参数
k_eff = 1.00202  # 有效增殖系数
k_inf = 1.00410  # 无穷增殖系数
epsilon = 0.001  # 临界判断阈值
a = 1  # 平板的宽度
D = 0.211e-2  # 扩散系数
v = 2.2e3  # 中子速度
L_square = 2.1037e-4  # 系统临界时的扩散长度
B_square = (np.pi / a) ** 2  # 系统临界时的几何曲率
l = 6  # 神经网络的深度
s = 20  # 神经网络的中间层隐藏神经单元数量
Pi = 100  # 初值权重
Pb = 100  # 边界权重
Pc = 100  # 额外配置点权重
C = 0.5  # 解析解参数
k_n = 1.0
l_n = 1.0
num_terms = 10  # 考虑的项数


# 定义初值条件
def phi_0(x):
    return np.cos(np.pi * x[:, 0:1] / a) - 0.4 * np.cos(2 * np.pi * x[:, 0:1] / a) - 0.4


# 解析解的计算
def phi_analytical(x):
    summation = 0
    B_n = lambda n: (2 * n - 1) * np.pi / a
    for n in range(1, 10):  # 选择一个足够大的上限来近似无穷级数
        l_n = L_square / (D * v * (1 + L_square * B_n(n) ** 2))
        k_n = k_inf / (1 + L_square * B_n(n) ** 2)
        phi_0 = lambda x_prime: np.cos(np.pi * x_prime / a) - 0.4 * np.cos(2 * np.pi * x_prime / a) - 0.4
        integrand = lambda x_prime: phi_0(x_prime) * np.cos(B_n(n) * x_prime)
        integral_result, _ = quad(integrand, -a / 2, a / 2)
        summation += (2 / a) * integral_result * np.cos(B_n(n) * x[:, 0:1]) * np.exp((k_n - 1) * x[:, 1:2] / l_n)

    return summation


# 定义几何网格
geom = dde.geometry.Interval(-a / 2, a / 2)
timedomain = dde.geometry.TimeDomain(0, 0.01)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# 定义微分方程
def pde(x, phi):
    dphi_xx = dde.grad.hessian(phi, x, i=0, j=0)
    dphi_t = dde.grad.jacobian(phi, x, i=0, j=1)
    return 1 / (D * v) * dphi_t - dphi_xx - (k_inf / k_eff - 1) / L_square * phi


# 定义边界条件
# bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 定义初值条件
ic = dde.IC(geomtime, lambda x: phi_0(x), lambda _, on_initial: on_initial)
# 定义数据
data = dde.data.TimePDE(
    geomtime, pde, [ic], num_domain=1000, num_boundary=200, num_initial=200, solution=phi_analytical)
# 定义神经网络
layer_size = [2] + [s] * l + [1]
activation = "tanh"
# 网络初始值权重采用高斯分布随机采样
initializer = "Glorot uniform"
net = dde.nn.PFNN(layer_size, activation, initializer)


# 定义硬边界条件
def output_transform(x, y):
    return (x[:, 0:1] + a / 2) * (x[:, 0:1] - a / 2) * y


net.apply_output_transform(output_transform)
# 定义模型
model = dde.Model(data, net)
# 定义求解器
model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, Pi])
model.train(epochs=2000)
# model.compile("L-BFGS")
# model.train()

X = geomtime.random_points(10000)
err = 1
# 创建一个空列表来存储 X[x_id] 的值
stored_points = []
while err > 0.02:
    f = model.predict(X, operator=pde)
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))
    # x_id = np.argmax(err_eq)
    # 检查 err_eq 的长度是否足够大
    if len(err_eq) >= 10:
        # 使用argsort获取降序排列的索引
        sorted_indices = np.argsort(err_eq[:, 0])[::-1]

        # 获取前10个最大值的索引
        top_indices = sorted_indices[:10]
        # 将对应索引的点添加到列表中
        for x_id in top_indices:
            stored_points.append(X[x_id].flatten())
            print("Adding new point:", X[x_id], "\n")
            data.add_anchors(X[x_id])
    else:
        print("Error: Not enough elements in err_eq to get top 10 indices.")
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"], loss_weights=[1, Pi])
    losshistory, train_state = model.train(iterations=1000, disregard_previous_best=True, callbacks=[early_stopping])
    # model.compile("L-BFGS")
    # losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# ---------------------------------------------------------------------------------------------
# 可视化
# 创建保存图像的文件夹
# output_folder = "figure/算例4"
# os.makedirs(output_folder, exist_ok=True)

# 设置中文字体

# 替换为您的字体文件路径
font_path = '/System/Library/Fonts/STHeiti Light.ttc'

# 添加字体路径
font_properties = FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_properties.get_name()]
plt.rcParams['axes.unicode_minus'] = False


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
    plt.figure(figsize=(6, 5))  # 设置图像大小
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

# 2D
elif train_state.X_test.shape[1] == 2:
    for i in range(y_dim):
        plt.figure(figsize=(8, 6))  # 调整图像大小
        ax = plt.axes(projection=Axes3D.name)

        # 绘制所有测试点
        ax.plot3D(
            train_state.X_test[:, 0],
            train_state.X_test[:, 1],
            best_y[:, i],
            ".",
            label='初始训练点'
        )

        # 绘制存储点
        stored_points_array = np.array(stored_points)
        if len(stored_points_array) > 0:
            stored_points_predictions = model.predict(stored_points_array)
            ax.scatter(
                stored_points_array[:, 0],
                stored_points_array[:, 1],
                stored_points_predictions[:, i],  # 使用存储点的索引获取对应的神经网络预测值
                color='red',
                label='RAR加密点',
                s=8,
                alpha=0.7
            )

        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_zlabel("$\phi(x, t)$")

        # 移除网格
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)

        # 设置z轴坐标范围
        # ax.set_zlim(-0.05, 0.4)
        # 隐藏x轴标签
        # ax.set_xticks([])

        # 自定义视角
        ax.view_init(elev=30, azim=225)  # 仰角为0度，方位角为180度

        # 显示图例
        ax.legend()

        plt.tight_layout()  # 调整图像布局，确保坐标轴标签不重叠
        plt.tight_layout()  # 调整图像布局，确保坐标轴标签不重叠
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
plt.show()
