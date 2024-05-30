import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

n_episode = 1000
window_size = 10
reward_ddpg = moving_average(np.load('Reward_DDPG1000.npy'), window_size)
Sum_rate_ddpg = moving_average(np.load('Sum_V2I_rate_DDPG1000.npy'), window_size)
V2I_AoI_ddpg = moving_average(np.load('Sum_V2I_AoI_DDPG1000.npy'), window_size)
reward_td3 = moving_average(np.load('Reward_TD31000.npy'), window_size)
Sum_rate_td3 = moving_average(np.load('Sum_V2I_rate_TD31000.npy'), window_size)
V2I_AoI_td3 = moving_average(np.load('Sum_V2I_AoI_TD31000.npy'), window_size)
reward_ppo = moving_average(np.load('Reward_PPO1000.npy'), window_size)
Sum_rate_ppo = moving_average(np.load('Sum_V2I_rate_PPO1000.npy'), window_size)
V2I_AoI_ppo = moving_average(np.load('Sum_V2I_AoI_PPO1000.npy'), window_size)
reward_sac = moving_average(np.load('Reward_SAC1000.npy'), window_size)
Sum_rate_sac = moving_average(np.load('Sum_V2I_rate_SAC1000.npy'), window_size)
V2I_AoI_sac = moving_average(np.load('Sum_V2I_AoI_SAC1000.npy'), window_size)
reward_random = moving_average(np.load('Reward_Random1000.npy'), window_size)
Sum_rate_random = moving_average(np.load('Sum_V2I_rate_Random1000.npy'), window_size)
V2I_AoI_random = moving_average(np.load('Sum_V2I_AoI_Random1000.npy'), window_size)
reward_no_ris = moving_average(np.load('Reward_NO_RIS1000.npy'), window_size)
Sum_rate_no_ris = moving_average(np.load('Sum_V2I_rate_NO_RIS1000.npy'), window_size)
V2I_AoI_no_ris = moving_average(np.load('Sum_V2I_AoI_NO_RIS1000.npy'), window_size)

a = int(n_episode/4)

x1 =np.linspace(0,n_episode, n_episode, dtype=int)
#Reward 横坐标
x2 = np.linspace(0, n_episode-9, n_episode-9, dtype=int)
y = [-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]



plt.figure(1)
plt.plot(x2, reward_sac, color='red', label='SAC')
plt.plot(x2, reward_ppo, color='orange', label='PPO')
plt.plot(x2, reward_td3, color='green', label='TD3')
plt.plot(x2, reward_ddpg, color='blue', label='DDPG')
plt.plot(x2, reward_random, color='indigo', label='Random RIS Random RA')
plt.plot(x2, reward_no_ris, color='violet', label='NO RIS Random RA')
plt.grid(True, linestyle='-', linewidth=0.5)
# plt.yticks(y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='lower right', fontsize=8)
plt.savefig('D:\latex\projects\qkw1\Reward.pdf', dpi=300, format='pdf')


plt.figure(2)
plt.plot(x2, Sum_rate_sac, color='red', label='SAC')
plt.plot(x2, Sum_rate_ppo, color='orange', label='PPO')
plt.plot(x2, Sum_rate_td3, color='green', label='TD3')
plt.plot(x2, Sum_rate_ddpg, color='blue', label='DDPG')
plt.plot(x2, Sum_rate_random, color='indigo', label='Random_RIS Random_RA')
plt.plot(x2, Sum_rate_no_ris, color='violet', label='NO_RIS Random_RA')

plt.xlabel('Episode')
plt.ylabel('Sum rate')
plt.legend(loc='upper left')
plt.grid(True, linestyle='-', linewidth=0.5)


plt.figure(3)
plt.plot(x2, V2I_AoI_sac, color='red', label='SAC')
plt.plot(x2, V2I_AoI_ppo, color='orange', label='PPO')
plt.plot(x2, V2I_AoI_td3, color='green', label='TD3')
plt.plot(x2, V2I_AoI_ddpg, color='blue', label='DDPG')
plt.plot(x2, V2I_AoI_random, color='indigo', label='Random_RIS Random_RA')
plt.plot(x2, V2I_AoI_no_ris, color='violet', label='NO_RIS Random_RA')

plt.xlabel('Episode')
plt.ylabel('V2I AoI')
plt.legend(loc='upper right')
plt.grid(True, linestyle='-', linewidth=0.5)

plt.show()
