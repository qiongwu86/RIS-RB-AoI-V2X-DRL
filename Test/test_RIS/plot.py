import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
# 示例数据

x = [10, 40, 80, 120]
y1 = [59.448, 60.482, 61.354, 61.916]
y2 = [57.44, 58.174, 58.88, 59.254]
y3 = [55.62, 55.64, 56.07, 56.45]
y4 = [51.47, 51.75, 51.97, 52.24]
y5 = [42.08, 42.63, 43.21, 43.59]


z1 = [5.408, 5.352, 5.348, 5.326]
z2 = [9.122, 8.828, 8.59, 8.364]
z3 = [5.74, 5.74, 5.64, 5.66]
z4 = [7, 6.84, 7.15, 7.28]
z5 = [5.75, 5.79, 5.92, 5.96]

h1 = [14.882, 15.1205, 15.3385, 15.479]
h2 = [14.86175, 15.0285, 15.139, 15.2165]
h3 = [14.521, 14.581, 14.6895, 14.80516667]

'''x = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
y1 = [14.58, 14.584, 14.576, 14.654, 7.116, 6.542, 6.108, 5.788, 5.51, 5.37, 5.336]
y2 = [16.306, 15.756, 15.382, 15.284, 7.652, 6.854, 6.17, 5.802, 5.548, 5.456, 5.376]
y3 = [14.504, 13.512, 12.8, 12.394, 11.596, 11.044, 10.38, 10.076, 9.602, 9.42, 9.144]
y4 = [25.2, 25.1, 24.73, 24.6, 25.56, 20.84, 18.45, 15.94, 11.36, 7.19, 5.89]
y5 = [24.16, 25.2, 25.45, 23.31, 12.86, 10.55, 9.68, 9.18, 8.01, 7.54, 7.03]
y6 = [16.25, 14.91, 14.41, 14.04, 11.21, 10.39, 10.08, 8.99, 8.53, 7.01, 5.74]
y7 = [15.55, 14.88, 14.23, 13.32, 10.66, 9.7, 9.52, 8.94, 8.28, 6.82, 5.38]


z1 = [36.634, 38.992, 41.374, 43.688, 46.636, 48.996, 51.422, 53.828, 56.124, 58.464, 60.516]
z2 = [35.582, 37.938, 40.288, 42.636, 45.522, 47.95, 50.42, 52.71, 55.094, 57.356, 59.478]
z3 = [35.272, 37.614, 39.898, 42.188, 44.458, 46.794, 49.046, 51.052, 53.006, 55.164, 57.342]
z4 = [30.64, 32.79, 34.81, 36.59, 38.26, 40.73, 42.24, 43.96, 46.23, 47.99, 49.5]
z5 = [33.99, 36.06, 37.67, 39.65, 41.74, 43.45, 45.39, 46.96, 48.38, 49.87, 51.22]
z6 = [30.48, 32.15, 33.41, 34.37, 34.36, 33.79, 33.77, 35.19, 37.42, 39.76, 42.09]
z7 = [30.57, 31.96, 33.29, 34.08, 34.37, 33.91, 33.64, 34.98, 37.28, 39.61, 42.01]'''

# 创建一个图形对象和一个子图
# 绘制折线图
plt.figure(1)
plt.plot(x, y1, 'o-', markersize=5, label='SAC')
plt.plot(x, y2, '^-', markersize=5, label='PPO')
plt.plot(x, y3, 's-', markersize=5, label='TD3')
plt.plot(x, y4, 'p-', markersize=5, label='DDPG')
plt.plot(x, y5, '*-', markersize=5, label='Random RIS Random RA')
plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('The number of RIS elements')
plt.ylabel('Sum of V2I Rates')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(bbox_to_anchor=(0.34,0.3), fontsize=8)
plt.savefig('D:\latex\projects\qkw1\RIS-rate.pdf', dpi=300, format='pdf')
plt.figure(2)
plt.plot(x, z1, 'o-', markersize=5, label='SAC')
plt.plot(x, z2, '^-', markersize=5, label='PPO')
plt.plot(x, z3, 's-', markersize=5, label='TD3')
plt.plot(x, z4, 'p-', markersize=5, label='DDPG')
plt.plot(x, z5, '*-', markersize=5, label='Random RIS Random RA')

plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('The number of RIS elements')
plt.ylabel('Sum of V2I AoI')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(bbox_to_anchor=(0.34,0.8), fontsize=8)
plt.savefig('D:\latex\projects\qkw1\RIS-AoI.pdf', dpi=300, format='pdf')
plt.figure(3)
plt.plot(x, h1, 'o-', markersize=5, label='Users=4')
plt.plot(x, h2, '^-', markersize=5, label='Users=8')
plt.plot(x, h3, 's-', markersize=5, label='Users=12')
plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('The number of RIS elements')
plt.ylabel('Average V2I rate per user')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(loc='upper left', fontsize=10)

plt.savefig('D:\latex\projects\qkw1\RIS-rate-user.pdf', dpi=300, format='pdf')
# 显示图表
plt.show()
