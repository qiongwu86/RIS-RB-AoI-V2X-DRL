import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

def average(numbers):
    total = sum(numbers)
    length = len(numbers)
    average = total / length
    return average

n_episode = 30
window_size = 1
Sum_rate_10 = average(np.load('Sum_V2I_rate_SAC_test10.npy') )
V2I_AoI_10 = average(np.load('Sum_V2I_AoI_SAC_test10.npy'))

Sum_rate_30 = average(np.load('Sum_V2I_rate_SAC_test30.npy'))
V2I_AoI_30 = average(np.load('Sum_V2I_AoI_SAC_test30.npy'))

Sum_rate_50 = average(np.load('Sum_V2I_rate_SAC_test50.npy'))
V2I_AoI_50 = average(np.load('Sum_V2I_AoI_SAC_test50.npy'))

Sum_rate_70 = average(np.load('Sum_V2I_rate_SAC_test70.npy'))
V2I_AoI_70 = average(np.load('Sum_V2I_AoI_SAC_test70.npy'))

Sum_rate_90 = average(np.load('Sum_V2I_rate_SAC_test90.npy'))
V2I_AoI_90 = average(np.load('Sum_V2I_AoI_SAC_test90.npy'))
a = int(n_episode/4)

x1 =np.linspace(0,n_episode-1, n_episode, dtype=int)
#Reward 横坐标
x2 = np.linspace(0, n_episode-1, n_episode, dtype=int)

plt.figure(1)
plt.bar(x2, Sum_rate_10, color = 'red', label = 'Velocity_10')
plt.bar(x2, Sum_rate_30, color = 'orange', label = 'Velocity_30')
plt.bar(x2, Sum_rate_50, color = 'yellow', label = 'Velocity_50')
plt.bar(x2, Sum_rate_70, color = 'green', label = 'Velocity_70')
plt.bar(x2, Sum_rate_90, color = 'blue', label = 'Velocity_90')

plt.xlabel('Episode')
plt.ylabel('Sum rate')
plt.legend(loc = 'upper left')

plt.figure(3)
plt.bar(x2, V2I_AoI_10, color = 'red', label = 'Velocity_10')
plt.bar(x2, V2I_AoI_30, color = 'orange', label = 'Velocity_30')
plt.bar(x2, V2I_AoI_50, color = 'yellow', label = 'Velocity_50')
plt.bar(x2, V2I_AoI_70, color = 'green', label = 'Velocity_70')
plt.bar(x2, V2I_AoI_90, color = 'blue', label = 'Velocity_90')

plt.xlabel('Episode')
plt.ylabel('V2I AoI')
plt.legend(loc = 'upper right')


plt.show()