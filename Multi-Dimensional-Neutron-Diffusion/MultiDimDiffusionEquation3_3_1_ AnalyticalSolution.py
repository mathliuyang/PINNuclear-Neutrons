import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 模型方程的参数
v = 2.2e3  # m/s
D = 0.211e-2  # m
L_square = 2.1037e-4  # m^2
a = 1  # m
k_infinity = 1.0041


# 初始条件
def phi_0(x):
    return np.cos(np.pi * x / a) - 0.4 * np.cos(2 * np.pi * x / a) - 0.4


# 解析解的计算
def phi(x, t):
    summation = 0
    B_n = lambda n: (2 * n - 1) * np.pi / a
    for n in range(1, 10):  # 选择一个足够大的上限来近似无穷级数
        l_n = L_square / (D * v * (1 + L_square * B_n(n) ** 2))
        k_n = k_infinity / (1 + L_square * B_n(n) ** 2)

        integrand = lambda x_prime: phi_0(x_prime) * np.cos(B_n(n) * x_prime)
        integral_result, _ = quad(integrand, -a / 2, a / 2)
        summation += (2 / a) * integral_result * np.cos(B_n(n) * x) * np.exp((k_n - 1) * t / l_n)

    return summation

# 创建保存图像的文件夹
output_folder = "figure/算例3"
os.makedirs(output_folder, exist_ok=True)

# 设置中文字体

# 替换为您的字体文件路径
font_path = '/System/Library/Fonts/STHeiti Light.ttc'

# 添加字体路径
font_properties = FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_properties.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 生成二维网格
x_values = np.linspace(-a / 2, a / 2, 100)
t_values = np.linspace(0, 0.01, 100)
x_mesh, t_mesh = np.meshgrid(x_values, t_values)

# 计算解在二维网格上的值
phi_values = phi(x_mesh, t_mesh)

plt.show()
# 绘制三维图
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, t_mesh, phi_values, cmap='viridis', linewidth=0.5,edgecolors='k')
# ax.set_title('解析解')
ax.set_xlabel('x (m)')
ax.set_ylabel('t (s)')
ax.set_zlabel('$\phi(x, t)$')
# 自定义视角
ax.view_init(elev=30, azim=235)  # 仰角为30度，方位角为45度
# 添加颜色条
# colorbar = fig.colorbar(surf, ax=ax, pad=0.1)
# colorbar.set_label('$\phi(x, t)$', rotation=270, labelpad=15)

# 保存为png
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "AnalyticalSolution1.png"), format="png", dpi=200)
plt.show()