import numpy as np
import os
import scipy.io
import Environment3
import matplotlib.pyplot as plt
IS_TRAIN = 1

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
env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor, V2I_min)
env.new_random_game()  # initialize parameters in env

n_episode = 50
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(env.get_state()) * n_veh
n_output = 2 * n_RB + n_veh  # channel selection, power, phase
# --------------------------------------------------------------
## Let's go
if IS_TRAIN:
    # agent.load_models()
    Sum_rate_list = []
    Sum_AoI_list = []
    V2V_success_list = []
    record_reward_average = []
    cumulative_reward = 0
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

        env.renew_positions()  # update vehicle position
        env.renew_neighbor()

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        V2V_success_per_episode = []
        average_reward = 0
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = np.random.randint(-1, 1, n_output)
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
            train_reward, V2I_Rate, V2V_Rate, V2V_success, V2I_AoI = env.act_for_training1(action_channel, action_phase)

            Sum_rate_per_episode.append(((np.sum(V2I_Rate))))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))
            V2V_success_per_episode.append(V2V_success)

            record_reward[i_step] = train_reward

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
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('step:', i_episode, 'reward', average_reward)
        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

    print('-------- RANDOM -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(Sum_rate_list), 2), 'Mbps')
    print('Sum V2I AoI:', round(np.average(Sum_AoI_list), 2))
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

    '''x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
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
    plt.show()'''

    '''np.save('Test/Sum_V2I_rate_Random1000.npy', Sum_rate_list)
    np.save('Test/Sum_V2I_AoI_Random1000.npy', Sum_AoI_list)
    np.save('Test/Reward_Random1000.npy', record_reward_average)
    print('Training Done. Saving models...')'''
