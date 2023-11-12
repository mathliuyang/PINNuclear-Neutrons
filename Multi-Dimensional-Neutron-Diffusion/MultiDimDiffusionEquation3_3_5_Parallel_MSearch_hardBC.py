
import deepxde as dde
import numpy as np
import multiprocessing

# 初始化参数
k_eff = 1.00  # 有效增殖系数
k_inf = 1.0041  # 无穷增殖系数
epsilon = 0.001  # 临界判断阈值
a = 1  # 平板的宽度
D = 0.211e-2  # 扩散系数
v = 2.2e3  # 中子速度
L2 = 2.1037e-4  # 系统临界时的扩散长度
B2 = (np.pi / a) ** 2  # 系统临界时的几何曲率
l = 5  # 神经网络的深度
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
# bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
# 定义初值条件
ic = dde.IC(geomtime, lambda x: phi_0(x), lambda _, on_initial: on_initial)
# 定义数据
data = dde.data.TimePDE(geomtime, pde, [ic], num_domain=300, num_boundary=200, num_initial=1000, solution=None,
                        num_test=10000)
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
model.compile("adam", lr=0.001, loss_weights=[1, Pi])


# 定义一个函数，用于并行执行每个k_eff值的训练和处理
def process_k_eff(k_eff):
    # 训练模型
    losshistory, train_state = model.train(epochs=1000)
    observe_x = np.linspace(-a / 2, a / 2, 10)
    result = []
    for x_value in observe_x:
        t_value = 0.01
        dt = 0.0001
        result.append((model.predict(np.array([[x_value, t_value]])) - model.predict(
            np.array([[x_value, t_value - dt]]))) / dt)
    mean_result = sum(result) / len(result)
    return k_eff, mean_result, losshistory, train_state


if __name__ == '__main__':
    m = 10  # 将 k_eff 分为 m 份
    k_eff_min = 0.9  # 最小可能的 k_eff
    k_eff_max = 1.1  # 最大可能的 k_eff
    critical_k_eff = None  # 用于存储临界 k_eff

    # 创建多个进程池
    num_processes = multiprocessing.cpu_count()  # 获取可用的CPU核心数
    pool = multiprocessing.Pool(num_processes)

    while True:
        k_eff_values = np.linspace(k_eff_min, k_eff_max, m)
        mean_result = []

        # 并行执行每个k_eff值的训练和处理
        results = pool.map(process_k_eff, k_eff_values)

        for k_eff, result, losshistory, train_state in results:
            mean_result.append(result)
            if epsilon > result > -epsilon:
                critical_k_eff = k_eff
                print("Critical k_eff found:", critical_k_eff)
                print("mean_result:", result)
                dde.saveplot(losshistory, train_state, issave=True, isplot=True)
                break

        if critical_k_eff is not None:
            break

        min_negative = None
        max_positive = None
        min_found = False
        max_found = False

        for i in range(len(k_eff_values)):
            if mean_result[i] < -epsilon:
                if not min_found or k_eff_values[i] < min_negative:
                    min_negative = k_eff_values[i]
                    min_found = True
            if mean_result[i] > epsilon:
                if not max_found or k_eff_values[i] > max_positive:
                    max_positive = k_eff_values[i]
                    max_found = True

        if min_found:
            k_eff_max = min_negative
        if max_found:
            k_eff_min = max_positive

        print("mean_result:", mean_result)
        print("mean_result【-1】:", mean_result[-1])
        print("k_max:", k_eff_max)
        print("k_min:", k_eff_min)
        # 如果没有找到负值部分最小 k_eff 或正值部分最大 k_eff，结束循环
        if not min_found and not max_found:
            break
