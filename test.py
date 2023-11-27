import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def Median(array):
    '''
    Input an array and return the median number of the array
    Args:
        array: an array with n numbers
    Returns:
        array[m_index]: the median nubmer
            m_index: the index of the median number in array
    '''
    n = len(array) 
    m_index = n // 2
    array = sorted(array)
    return array[m_index]


# 示例数据点（您可以用实际数据替换这些数据点）
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([25, 9, 4, 1, 16, 0])
print(y[1:3])


# 创建三次样条插值对象
cs = CubicSpline(x, y)

# 生成用于绘图的细分数据点
x_fine = np.linspace(0, 6, 500)
y_fine = cs(x_fine)

# # 绘制原始数据点和插值曲线
# plt.plot(x, y, 'o', label='data points')
# plt.plot(x_fine, y_fine, label='cubic spline')
# plt.legend()
# plt.show()

