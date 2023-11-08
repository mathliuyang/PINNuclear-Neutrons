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
import numpy as np

# 定义各个区域的边界和点的数量
# 算例 1
num_points_D0 = 2000
num_points_DC1 = 0
num_points_DC2 = 0
num_points_DB1 = 0
num_points_DB2 = 0
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
# num_points_D0 = 1000
# num_points_DC1 = 0
# num_points_DC2 = 500
# num_points_DB1 = 250
# num_points_DB2 = 250

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
x = np.concatenate((x_D0, x_DC1, x_DC2, x_DB1, x_DB2))
t = np.concatenate((t_D0, t_DC1, t_DC2, t_DB1, t_DB2))

# 打印生成的点的数量
print(f"Total Points: {len(x)}")

import matplotlib.pyplot as plt

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

