import numpy as np
import os
import scipy.io
import Environment3
from RL_train3 import Agent
from RL_train4 import Agent_TD3
from RL_train5 import SAC_Trainer
from RL_train5 import ReplayBuffer
from RL_train6 import PPO
from RL_train6 import Memory
import Environment4
import matplotlib.pyplot as plt

# ################## SETTINGS ######################
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
width = 750 / 2
height = 1298 / 2


position_RIS = [290, 375]
BS_position = [180, 270]

RIS_adjust_phase = 8
max_power = 23
min_power = 1
n_veh = 4
n_neighbor = 1
n_RB = n_veh
V2I_min = 3.16
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------- characteristics related to the network -------- #
batch_size = 64
memory_size = 1000000


env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor, V2I_min)
env.new_random_game()  # initialize parameters in env

n_episode = 1000
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 50  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(env.get_state()) * n_veh
n_output = 2*n_RB + n_veh  # channel selection, power, phase
# --------------------------------------------------------------
label_sac = 'model/sac_model'
model_path_sac = label_sac + '/agent'
label_ppo = 'model/ppo_model'
model_path_ppo = label_ppo + '/agent'
replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
action_range = 1.0
AUTO_ENTROPY=True
DETERMINISTIC=False
RL_sac = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)
#---------------------------------------------------------------
update_timestep = 100  # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor
lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)
memory = Memory()
RL_ppo = PPO(n_input, n_output, action_std, lr, betas, gamma, K_epochs, eps_clip)
#---------------------------------------------------------------
fc1_dims = 512
fc2_dims = 512
fc3_dims = 512
fc4_dims = 512
alpha = 0.0001
beta = 0.001
tau = 0.005
RL_ddpg = Agent(alpha, beta, int(n_input/4), tau, n_output, gamma, memory_size,
                fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, n_veh)
#----------------------------------------------------------------
RL_td3 = Agent_TD3(alpha, beta, n_input, tau, gamma, 12, memory_size,
                   fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, 2, 'OU')


## Let's go
def sac_test():
    print("\nRestoring the sac model...")
    RL_sac.load_model(model_path_sac)

    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    '''Vehicle_positions_x4 = []
    Vehicle_positions_y4 = []
    Vehicle_positions_x5 = []
    Vehicle_positions_y5 = []
    Vehicle_positions_x6 = []
    Vehicle_positions_y6 = []
    Vehicle_positions_x7 = []
    Vehicle_positions_y7 = []
    Vehicle_positions_x8 = []
    Vehicle_positions_y8 = []
    Vehicle_positions_x9 = []
    Vehicle_positions_y9 = []
    Vehicle_positions_x10 = []
    Vehicle_positions_y10 = []
    Vehicle_positions_x11 = []
    Vehicle_positions_y11 = []'''
    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16) * 100

        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])
        '''Vehicle_positions_x4.append(env.vehicles[4].position[0])
        Vehicle_positions_y4.append(env.vehicles[4].position[1])
        Vehicle_positions_x5.append(env.vehicles[5].position[0])
        Vehicle_positions_y5.append(env.vehicles[5].position[1])
        Vehicle_positions_x6.append(env.vehicles[6].position[0])
        Vehicle_positions_y6.append(env.vehicles[6].position[1])
        Vehicle_positions_x7.append(env.vehicles[7].position[0])
        Vehicle_positions_y7.append(env.vehicles[7].position[1])
        Vehicle_positions_x8.append(env.vehicles[8].position[0])
        Vehicle_positions_y8.append(env.vehicles[8].position[1])
        Vehicle_positions_x9.append(env.vehicles[9].position[0])
        Vehicle_positions_y9.append(env.vehicles[9].position[1])
        Vehicle_positions_x10.append(env.vehicles[10].position[0])
        Vehicle_positions_y10.append(env.vehicles[10].position[1])
        Vehicle_positions_x11.append(env.vehicles[11].position[0])
        Vehicle_positions_y11.append(env.vehicles[11].position[1])'''

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = RL_sac.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(
                        np.clip(((action[1 + i * 2] + 1) / 2) * max_power, min_power, max_power))  # power selected by PL
            for n in range(n_veh):
                action_all_training_phase[n] = ((action[2 * n_RB + n] + 1) / 2) * RIS_adjust_phase
            action_channel = action_all_training.copy()
            action_phase = action_all_training_phase.copy()
            # 在act_for_training_ddpg中编写每个车的奖励算法，然后求平均
            V2I_Rate, V2V_Rate, V2I_AoI, V2V_success = env.act_for_testing(action_channel, action_phase)
            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            env.renew_channel_fastfading()

            env.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))

        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))
    plt.figure(1)
    #fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_position[0], BS_position[1], 'o', markersize=5, color='black', label='BS')
    plt.plot(position_RIS[0], position_RIS[1], 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0[0], Vehicle_positions_y0[1], '>', markersize=5, color='red')
    plt.plot(Vehicle_positions_x1[0], Vehicle_positions_y1[1], 'v', markersize=5, color='orange')
    plt.plot(Vehicle_positions_x2[0], Vehicle_positions_y2[1], '^', markersize=5, color='green')
    plt.plot(Vehicle_positions_x3[0], Vehicle_positions_y3[1], 's', markersize=5, color='blue')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0)
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1)
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2)
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3)
    '''plt.plot(Vehicle_positions_x4, Vehicle_positions_y4)
    plt.plot(Vehicle_positions_x5, Vehicle_positions_y5)
    plt.plot(Vehicle_positions_x6, Vehicle_positions_y6)
    plt.plot(Vehicle_positions_x7, Vehicle_positions_y7)
    plt.plot(Vehicle_positions_x8, Vehicle_positions_y8)
    plt.plot(Vehicle_positions_x9, Vehicle_positions_y9)
    plt.plot(Vehicle_positions_x10, Vehicle_positions_y10)
    plt.plot(Vehicle_positions_x11, Vehicle_positions_y11)'''

    # 添加图例
    plt.legend(loc='upper left')

    plt.show()
    '''x = np.linspace(0, n_episode_test - 1, n_episode_test, dtype=int)
    y1 = Sum_rate_list
    y2 = Sum_AoI_list
    plt.figure(1)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Sum V2I rate')

    plt.figure(2)
    plt.plot(x, y2)
    plt.xlabel('Episode')
    plt.ylabel('Sum V2I AoI')
    plt.show()'''

    print('-------- SAC -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    average_AoI = round(np.average(Sum_AoI_list), 2)
    average_rate = round(np.average(Sum_rate_list), 2)
    average_pr = round(np.average(V2V_success_list), 4)

    with open("Data.txt", "a") as f:
        f.write('-------- SAC ------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(Sum_rate_list), 5)) + ' Mbps\n')
        f.write('Sum V2I AoI: ' + str((np.average(Sum_AoI_list))) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')

    return average_AoI, average_rate, average_pr

def ppo_test():
    print("\nRestoring the ppo model...")
    RL_ppo.load_model(model_path_ppo)

    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16) * 100

        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = RL_ppo.select_action(np.asarray(state_old_all).flatten(), memory)
            action = np.clip(action, -0.999, 0.999)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(
                        np.clip(((action[1 + i * 2] + 1) / 2) * max_power, 1, max_power))  # power selected by PL
            for n in range(n_veh):
                action_all_training_phase[n] = ((action[2 * n_RB + n] + 1) / 2) * RIS_adjust_phase
            action_channel = action_all_training.copy()
            action_phase = action_all_training_phase.copy()
            # 在act_for_training_ddpg中编写每个车的奖励算法，然后求平均
            V2I_Rate, V2V_Rate, V2I_AoI, V2V_success = env.act_for_testing(action_channel, action_phase)
            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            env.renew_channel_fastfading()

            env.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))

        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_position[0], BS_position[1], 'o', markersize=5, color='black', label='BS')
    plt.plot(position_RIS[0], position_RIS[1], 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0[0], Vehicle_positions_y0[1], '>', markersize=5, color='red')
    plt.plot(Vehicle_positions_x1[0], Vehicle_positions_y1[1], 'v', markersize=5, color='orange')
    plt.plot(Vehicle_positions_x2[0], Vehicle_positions_y2[1], '^', markersize=5, color='green')
    plt.plot(Vehicle_positions_x3[0], Vehicle_positions_y3[1], 's', markersize=5, color='blue')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0, color='red', label='vehicle0')
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1, color='orange', label='vehicle1')
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2, color='green', label='vehicle2')
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3, color='blue', label='vehicle3')

    # 添加图例
    plt.legend(loc='upper left')

    plt.show()

    print('-------- PPO -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))
    average_AoI = round(np.average(Sum_AoI_list), 2)
    average_rate = round(np.average(Sum_rate_list), 2)
    average_pr = round(np.average(V2V_success_list), 4)


    with open("Data.txt", "a") as f:
        f.write('-------- PPO -------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(Sum_rate_list), 5)) + ' Mbps\n')
        f.write('Sum V2I AoI: ' + str((np.average(Sum_AoI_list))) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')

    return average_AoI, average_rate, average_pr

def td3_test():
    print("\nRestoring the td3 model...")
    RL_td3.load_models()

    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16) * 100

        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = RL_td3.choose_action(np.asarray(state_old_all).flatten())
            action = np.clip(action, -0.999, 0.999)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(
                        np.clip(((action[1 + i * 2] + 1) / 2) * max_power, 1, max_power))  # power selected by PL
            for n in range(n_veh):
                action_all_training_phase[n] = ((action[2 * n_RB + n] + 1) / 2) * RIS_adjust_phase
            action_channel = action_all_training.copy()
            action_phase = action_all_training_phase.copy()
            # 在act_for_training_ddpg中编写每个车的奖励算法，然后求平均
            V2I_Rate, V2V_Rate, V2I_AoI, V2V_success = env.act_for_testing(action_channel, action_phase)
            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            env.renew_channel_fastfading()

            env.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))

        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_position[0], BS_position[1], 'o', markersize=5, color='black', label='BS')
    plt.plot(position_RIS[0], position_RIS[1], 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0[0], Vehicle_positions_y0[1], '>', markersize=5, color='red')
    plt.plot(Vehicle_positions_x1[0], Vehicle_positions_y1[1], 'v', markersize=5, color='orange')
    plt.plot(Vehicle_positions_x2[0], Vehicle_positions_y2[1], '^', markersize=5, color='green')
    plt.plot(Vehicle_positions_x3[0], Vehicle_positions_y3[1], 's', markersize=5, color='blue')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0, color='red', label='vehicle0')
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1, color='orange', label='vehicle1')
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2, color='green', label='vehicle2')
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3, color='blue', label='vehicle3')

    # 添加图例
    plt.legend(loc='upper left')

    plt.show()

    print('-------- TD3 -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- TD3------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(Sum_rate_list), 5)) + ' Mbps\n')
        f.write('Sum V2I AoI: ' + str((np.average(Sum_AoI_list))) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
def ddpg_test():
    print("\nRestoring the ddpg model...")
    RL_ddpg.load_models()

    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16) * 100

        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = RL_ddpg.choose_action(np.asarray(state_old_all).flatten())
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(
                        np.clip(((action[1 + i * 2] + 1) / 2) * max_power, 1, max_power))  # power selected by PL
            for n in range(n_veh):
                action_all_training_phase[n] = ((action[2 * n_RB + n] + 1) / 2) * RIS_adjust_phase
            action_channel = action_all_training.copy()
            action_phase = action_all_training_phase.copy()
            # 在act_for_training_ddpg中编写每个车的奖励算法，然后求平均
            V2I_Rate, V2V_Rate, V2I_AoI, V2V_success = env.act_for_testing(action_channel, action_phase)
            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            env.renew_channel_fastfading()

            env.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))

        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_position[0], BS_position[1], 'o', markersize=5, color='black', label='BS')
    plt.plot(position_RIS[0], position_RIS[1], 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0[0], Vehicle_positions_y0[1], '>', markersize=5, color='red')
    plt.plot(Vehicle_positions_x1[0], Vehicle_positions_y1[1], 'v', markersize=5, color='orange')
    plt.plot(Vehicle_positions_x2[0], Vehicle_positions_y2[1], '^', markersize=5, color='green')
    plt.plot(Vehicle_positions_x3[0], Vehicle_positions_y3[1], 's', markersize=5, color='blue')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0, color='red', label='vehicle0')
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1, color='orange', label='vehicle1')
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2, color='green', label='vehicle2')
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3, color='blue', label='vehicle3')

    # 添加图例
    plt.legend(loc='upper left')

    plt.show()

    print('-------- DDPG -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- ddpg  -------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(Sum_rate_list), 5)) + ' Mbps\n')
        f.write('Sum V2I AoI: ' + str((np.average(Sum_AoI_list))) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
def random_test():
    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16) * 100

        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = np.random.randint(-1, 1, n_output)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(
                        np.clip(((action[1 + i * 2] + 1) / 2) * max_power, 1, max_power))  # power selected by PL
            for n in range(n_veh):
                action_all_training_phase[n] = ((action[2 * n_RB + n] + 1) / 2) * RIS_adjust_phase
            action_channel = action_all_training.copy()
            action_phase = action_all_training_phase.copy()
            # 在act_for_training_ddpg中编写每个车的奖励算法，然后求平均
            V2I_Rate, V2V_Rate, V2I_AoI, V2V_success = env.act_for_testing(action_channel, action_phase)
            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            env.renew_channel_fastfading()

            env.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))

        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_position[0], BS_position[1], 'o', markersize=5, color='black', label='BS')
    plt.plot(position_RIS[0], position_RIS[1], 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0[0], Vehicle_positions_y0[1], '>', markersize=5, color='red')
    plt.plot(Vehicle_positions_x1[0], Vehicle_positions_y1[1], 'v', markersize=5, color='orange')
    plt.plot(Vehicle_positions_x2[0], Vehicle_positions_y2[1], '^', markersize=5, color='green')
    plt.plot(Vehicle_positions_x3[0], Vehicle_positions_y3[1], 's', markersize=5, color='blue')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0, color='red', label='vehicle0')
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1, color='orange', label='vehicle1')
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2, color='green', label='vehicle2')
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3, color='blue', label='vehicle3')

    # 添加图例
    plt.legend(loc='upper left')

    plt.show()

    print('-------- RANDOM -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- random -------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(Sum_rate_list), 5)) + ' Mbps\n')
        f.write('Sum V2I AoI: ' + str((np.average(Sum_AoI_list))) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
def no_ris_test():
    env1 = Environment4.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor,
                                V2I_min)
    env1.new_random_game()  # initialize parameters in env

    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16) * 100

        env1.renew_positions()  # update vehicle position
        env1.renew_neighbor()

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env1.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []

        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            # receive observation
            action = np.random.randint(-1, 1, n_output)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(
                        np.clip(((action[1 + i * 2] + 1) / 2) * max_power, 1, max_power))  # power selected by PL
            action_channel = action_all_training.copy()
            # 在act_for_training_ddpg中编写每个车的奖励算法，然后求平均
            V2I_Rate, V2V_Rate, V2I_AoI, V2V_success = env1.act_for_testing(action_channel)
            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            env1.renew_channel_fastfading()

            env1.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env1.get_state([i, j])
                    state_new_all.append((state_new))

            # old observation = new_observation
            state_old_all = state_new_all

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))

        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

    print('-------- NO RIS -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- no ris -------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(Sum_rate_list), 5)) + ' Mbps\n')
        f.write('Sum V2I AoI: ' + str((np.average(Sum_AoI_list))) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
if __name__ == "__main__":
    sac_test()
    #ppo_test()
    #td3_test()
    #ddpg_test()
    #random_test()
    #no_ris_test()



