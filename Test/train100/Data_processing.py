import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

n_episode = 1000
window_size = 10
reward_sac = moving_average((np.load('Reward_SAC1000.npy')), window_size)
Sum_rate_sac = moving_average(np.load('Sum_V2I_rate_SAC1000.npy'), window_size)
V2I_AoI_sac = moving_average(np.load('Sum_V2I_AoI_SAC1000.npy'), window_size)

a = int(n_episode/4)

x1 =np.linspace(0,n_episode-1, n_episode, dtype=int)
#Reward 横坐标
x2 = np.linspace(0, n_episode-1, n_episode-9, dtype=int)

plt.figure(1)
plt.plot(x2, reward_sac, color = 'red', label = 'SAC')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc = 'lower right')

plt.figure(2)
plt.plot(x2, Sum_rate_sac, color = 'red', label = 'SAC')

plt.xlabel('Episode')
plt.ylabel('Sum rate')
plt.legend(loc = 'lower right')

plt.figure(3)
plt.plot(x2, V2I_AoI_sac, color = 'red', label = 'SAC')

plt.xlabel('Episode')
plt.ylabel('V2I AoI')
plt.legend(loc = 'upper right')


plt.show()