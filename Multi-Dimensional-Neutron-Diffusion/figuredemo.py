import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 替换为您的字体文件路径
font_path = '/System/Library/Fonts/STHeiti Light.ttc'

# 添加字体路径
font_properties = FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_properties.get_name()]

# 示例
plt.figure()
plt.plot([1, 2, 3], label="示例", color="tab:blue")
plt.title("示例标题")
plt.xlabel("X轴")
plt.ylabel("Y轴")
plt.legend()
plt.show()
