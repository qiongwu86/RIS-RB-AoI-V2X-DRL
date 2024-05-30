import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
# 示例数据
# 示例数据
x = [2, 4, 6, 8, 10]
y1 = [5.146, 5.228, 5.252, 5.376, 5.584]
y2 = [7.658, 8.48, 8.736, 9.144, 9.62]
y3 = [5.62, 5.71, 5.76, 5.76, 6.08]
y4 = [6.38, 6.5, 6.87, 7.03, 7.11]
y5 = [5.62, 5.66, 5.73, 5.74, 5.76]
y6 = [5.28, 5.34, 5.35, 5.38, 5.38]



z1 = [62.106, 61.344, 60.318, 59.478, 58.694]
z2 = [61.038, 59.404, 58.166, 57.342, 56.924]
z3 = [57.33, 56.88, 56.22, 55.63, 54.94]
z4 = [57.11, 54.85, 52.72, 51.22, 50.39]
z5 = [42.61, 42.42, 42.64, 42.09, 41.96]
z6 = [42.65, 42.29, 42.13, 42.01, 41.85]

h1 = [0.96156, 0.94732, 0.92624, 0.90946, 0.89202]
h2 = [0.931344, 0.88868, 0.85614, 0.83042, 0.81744]
h3 = [0.7193, 0.6566, 0.624, 0.5824, 0.5603]
h4 = [0.8119, 0.7612, 0.7118, 0.6806, 0.6542]
h5 = [0.2622, 0.2478, 0.2343, 0.2151, 0.2006]
h6 = [0.3274, 0.2828, 0.2618, 0.2481, 0.2274]


# 创建一个图形对象和一个子图
# 绘制折线图
plt.figure(1)
plt.plot(x, y1, 'o-', markersize=5, label='SAC')
plt.plot(x, y2, '^-', markersize=5, label='PPO')
plt.plot(x, y3, 's-', markersize=5, label='TD3')
plt.plot(x, y4, 'p-', markersize=5, label='DDPG')
plt.plot(x, y5, 'v-', markersize=5, label='Random RA Random RIS')
plt.plot(x, y6, '>-', markersize=5, label='Random RA NO RIS')
plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('V2V transmission payload size D (x 1060 Bytes)')
plt.ylabel('Sum of V2I AoI')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(loc='upper left', fontsize=7.2)
plt.savefig('D:\latex\projects\qkw1\Byte-AoI.pdf', dpi=300, format='pdf')
plt.figure(2)
plt.plot(x, z1, 'v-', markersize=5, label='SAC')
plt.plot(x, z2, '^-', markersize=5, label='PPO')
plt.plot(x, z3, 's-', markersize=5, label='TD3')
plt.plot(x, z4, 'p-', markersize=5, label='DDPG')
plt.plot(x, z5, 'o-', markersize=5, label='Random RA Random RIS')
plt.plot(x, z6, '>-', markersize=5, label='Random RA NO RIS')


plt.xticks(x)
# 设置图表标题和坐标轴标签
plt.xlabel('V2V transmission payload size D (x 1060 Bytes)')
plt.ylabel('Sum of V2I Rates')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(bbox_to_anchor=(0.34,0.4), fontsize=8)
plt.savefig('D:\latex\projects\qkw1\Byte-rate.pdf', dpi=300, format='pdf')
plt.figure(3)
plt.plot(x, h1, 'v-', markersize=5, label='SAC')
plt.plot(x, h2, '^-', markersize=5, label='PPO')
plt.plot(x, h3, 's-', markersize=5, label='TD3')
plt.plot(x, h4, 'p-', markersize=5, label='DDPG')
plt.plot(x, h5, 'o-', markersize=5, label='Random RA Random RIS')
plt.plot(x, h6, '>-', markersize=5, label='Random RA NO RIS')


plt.xticks(x)
plt.ylim(0, 1)
# 设置图表标题和坐标轴标签
plt.xlabel('V2V transmission payload size D (x 1060 Bytes)')
plt.ylabel('Average V2V payload transmission probability')
plt.grid(True, linestyle='-', linewidth=0.5)
# 添加图例
plt.legend(bbox_to_anchor=(0.34,0.5), fontsize=8)
plt.savefig('D:\latex\projects\qkw1\Byte-pr.pdf', dpi=300, format='pdf')

# 显示图表
plt.show()
