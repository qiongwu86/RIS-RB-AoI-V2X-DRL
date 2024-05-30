import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

# 示例数据
x = [13, 15, 17, 19, 21, 23]
#y1 = [6.542, 6.108, 5.788, 5.51, 5.37, 5.336]
y2 = [6.854, 6.17, 5.802, 5.548, 5.456, 5.376]
y3 = [11.044, 10.38, 10.076, 9.602, 9.42, 9.144]
y4 = [11.62, 9.72, 7.69, 6.62, 6.02, 5.76]
y5 = [10.55, 9.68, 9.18, 8.01, 7.54, 7.03]
y6 = [10.39, 10.08, 8.99, 8.53, 7.01, 5.74]
y7 = [9.7, 9.52, 8.94, 8.28, 6.82, 5.38]


#z1 = [48.996, 51.422, 53.828, 56.124, 58.464, 60.516]
z2 = [47.95, 50.42, 52.71, 55.094, 57.356, 59.478]
z3 = [46.794, 49.046, 51.052, 53.006, 55.164, 57.342]
z4 = [44.62, 46.9, 49.33, 51.6, 53.73, 55.63]
z5 = [43.45, 45.39, 46.96, 48.38, 49.87, 51.22]
z6 = [33.79, 33.77, 35.19, 37.42, 39.76, 42.09]
z7 = [33.91, 33.64, 34.98, 37.28, 39.61, 42.01]

#h1 = [0.93254, 0.93028, 0.9282, 0.92306, 0.91838, 0.90896]
h2 = [0.9321, 0.93102, 0.92698, 0.92324, 0.91754, 0.90946]
h3 = [0.8863, 0.88164, 0.86724, 0.84982, 0.8423, 0.83042]
h4 = [0.7666, 0.7631, 0.7627, 0.7574, 0.7474, 0.7346]
h5 = [0.778, 0.7678, 0.7455, 0.7267, 0.7016, 0.6806]
h6 = [0.3499, 0.2727, 0.2431, 0.2339, 0.2268, 0.2151]
h7 = [0.4183, 0.3246, 0.2846, 0.2714, 0.258, 0.2481]
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
#plt.plot(x, y1, 'o-', markersize=5, label='SAC N=40')
plt.plot(x, y2, 'v-', markersize=5, label='SAC')
plt.plot(x, y3, '^-', markersize=5, label='PPO')
plt.plot(x, y4, 's-', markersize=5, label='TD3')
plt.plot(x, y5, 'p-', markersize=5, label='DDPG')
plt.plot(x, y6, '*-', markersize=5, label='Random RIS Random RA')
plt.plot(x, y7, '+-', markersize=5, label='NO RIS Random RA')
plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('V2I power')
plt.ylabel('Sum of V2I AoI')

plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(loc='upper right', fontsize=8)
plt.savefig('D:\latex\projects\qkw1\Power-AoI.pdf', dpi=300, format='pdf')
plt.figure(2)
#plt.plot(x, z1, 'o-', markersize=5, label='SAC N=40')
plt.plot(x, z2, 'v-', markersize=5, label='SAC')
plt.plot(x, z3, '^-', markersize=5, label='PPO')
plt.plot(x, z4, 's-', markersize=5, label='TD3')
plt.plot(x, z5, 'p-', markersize=5, label='DDPG')
plt.plot(x, z6, '*-', markersize=5, label='Random RIS Random RA')
plt.plot(x, z7, '+-', markersize=5, label='NO RIS Random RA')

plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('V2I power')
plt.ylabel('Sum of V2I Rates')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(loc='upper left', fontsize=8)
plt.savefig('D:\latex\projects\qkw1\Power-rate.pdf', dpi=300, format='pdf')
plt.figure(3)
#plt.plot(x, h1, 'o-', markersize=5, label='SAC N=40')
plt.plot(x, h2, 'v-', markersize=5, label='SAC')
plt.plot(x, h3, '^-', markersize=5, label='PPO')
plt.plot(x, h4, 's-', markersize=5, label='TD3')
plt.plot(x, h5, 'p-', markersize=5, label='DDPG')
plt.plot(x, h6, '*-', markersize=5, label='Random RIS Random RA')
plt.plot(x, h7, '+-', markersize=5, label='NO RIS Random RA')

plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('V2I power')
plt.ylabel('Average V2V payload transmission probability')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(bbox_to_anchor=(1,0.5), fontsize=8)
plt.savefig('D:\latex\projects\qkw1\Power-pr.pdf', dpi=300, format='pdf')
# 显示图表
plt.show()
