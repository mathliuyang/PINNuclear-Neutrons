import numpy as np

# 创建一个NumPy数组
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9])

# 使用argsort获取数组索引，按值从大到小排序
sorted_indices = np.argsort(arr)[::-1]

# 获取前10个最大值的索引
top_10_indices = sorted_indices[:10]

print("数组:", arr)
print("前10个最大值的索引:", top_10_indices)
