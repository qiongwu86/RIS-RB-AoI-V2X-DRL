import numpy as np
import os
import scipy.io
import Environment4
import matplotlib.pyplot as plt
from RL_train5 import SAC_Trainer
from  RL_train5 import ReplayBuffer
import matplotlib.pyplot as plt

IS_TRAIN = 1
label = 'model/sac_model_no_ris'
model_path = label + '/agent'
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


BS_position = [180, 270]


max_power = 23
n_veh = 4
n_neighbor = 1
n_RB = n_veh
V2I_min = 3.16
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# ------- characteristics related to the network -------- #
batch_size = 64
# ------------------------------
env = Environment4.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor, V2I_min)
env.new_random_game()  # initialize parameters in env

n_episode = 1000
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(env.get_state()) * n_veh
n_output = 2 * n_RB  # channel selection, power, phase
action_range = 1.0
# --------------------------------------------------------------
#agent = SAC_Trainer(alpha, beta, n_input, tau, gamma, 12 ,memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, 2, 'OU')
replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
agent = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
frame_idx = 0
explore_steps = 0 # for random action sampling in the beginning of training
## Let's go
if IS_TRAIN:
    # agent.load_models()
    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    record_reward_average = []
    cumulative_reward = 0
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
    '''
    Vehicle_positions_x12 = []
    Vehicle_positions_y12 = []
    Vehicle_positions_x13 = []
    Vehicle_positions_y13 = []
    Vehicle_positions_x14 = []
    Vehicle_positions_y14 = []
    Vehicle_positions_x15 = []
    Vehicle_positions_y15 = []'''
    for i_episode in range(n_episode):
        done = 0
        print("-------------------------------------------------------------------------------------------------------")
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)
        record_AoI = np.zeros([n_RB, n_step_per_episode], dtype=np.float16)
        per_total_user = np.zeros([n_RB, n_step_per_episode], dtype=np.float16)

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')
        env.AoI = np.ones(env.n_Veh, dtype=np.float16)*100

        if i_episode % 20 == 0:
            env.renew_positions()  # update vehicle position
            env.renew_neighbor()
            env.overall_channel()
            env.renew_channel_fastfading()
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
        average_reward = 0
        V2V_success_per_episode = []
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            # receive observation
            if frame_idx > explore_steps:
                action = agent.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            else:
                action = agent.policy_net.sample_action()
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = np.round(np.clip(((action[1 + i * 2] + 1) / 2) * max_power, 1, max_power))  # power selected by PL
            action_channel = action_all_training.copy()
            train_reward, V2I_Rate, V2V_Rate, V2V_success, V2I_AoI = env.act_for_training1(action_channel)

            Sum_rate_per_episode.append(((np.sum(V2I_Rate))))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            record_reward[i_step] = train_reward

            env.renew_channel_fastfading()
            env.Compute_Interference(action_channel)

            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            # taking the agents actions, states and reward
            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                               train_reward, np.asarray(state_new_all).flatten(), done)

            # agents take random samples and learn
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    _ = agent.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                                     target_entropy=-1. * n_output)

            # old observation = new_observation
            state_old_all = state_new_all
            frame_idx += 1



        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        V2V_success_list.append(np.mean(V2V_success_per_episode))
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('step:', i_episode, 'reward', average_reward)
        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

        if (i_episode+1) % 100 == 0 and i_episode != 0:
            agent.save_model(model_path)

    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_position[0], BS_position[1], 'o', markersize=5, color='black', label='BS')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0)
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1)
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2)
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3)

    # 添加标题和坐标轴标签
    plt.title('Trajectory Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    print('-------- NO RIS -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))
    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    plt.figure(2)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    y2 = Sum_AoI_list
    plt.figure(3)
    plt.plot(x, y2)
    plt.xlabel('Episode')
    plt.ylabel('Sum AoI')
    y3 = Sum_rate_list
    plt.figure(4)
    plt.plot(x, y3)
    plt.xlabel('Episode')
    plt.ylabel('Sum rate')
    plt.show()

    np.save('Test/Sum_V2I_rate_SAC_NO_RIS1000.npy', Sum_rate_list)
    np.save('Test/Sum_V2I_AoI_SAC_NO_RIS1000.npy', Sum_AoI_list)
    np.save('Test/Reward_SAC_NO_RIS1000.npy', record_reward_average)
    print('Training Done. Saving models...')
