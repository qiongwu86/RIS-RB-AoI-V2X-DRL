#PPO
import numpy as np

import Environment3
from RL_train6 import PPO
from RL_train6 import Memory
import matplotlib.pyplot as plt

n_elements_total = 12  # number of elements
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
env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor, V2I_min)
env.new_random_game()  # initialize parameters in env

n_episode = 1000
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100  # test episodes
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(env.get_state()) * n_veh
n_output = 2 * n_RB + n_veh  # channel selection, power, phase

update_timestep = 10  # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 8  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)

# --------------------------------------------------------------
memory = Memory()
agent = PPO(n_input, n_output, action_std, lr, betas, gamma, K_epochs, eps_clip)

label = 'model/ppo_model8'
model_path = label + '/agent'

## Let's go
if IS_TRAIN:
    # agent.load_models()
    Sum_rate_list = []
    Sum_AoI_list = []
    record_reward_average = []
    cumulative_reward = 0
    time_step = 0
    for i_episode in range(n_episode):
        done = False
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
            env.overall_channel(env.theta, n_elements_total)
            env.renew_channel_fastfading()

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = env.get_state([i, j])
                state_old_all.append(state)

        Sum_rate_per_episode = []
        Sum_AoI_per_episode = []
        average_reward = 0
        for i_step in range(n_step_per_episode):
            time_step += 1
            state_new_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_phase = np.zeros([n_veh], dtype=int)  # phase
            # receive observation
            action = agent.select_action(np.asarray(state_old_all).flatten(), memory)
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
            train_reward, V2I_Rate, V2V_Rate, V2V_success, V2I_AoI = env.act_for_training1(action_channel, action_phase)

            Sum_rate_per_episode.append((np.sum(V2I_Rate)))
            Sum_AoI_per_episode.append((np.sum(V2I_AoI)))

            record_reward[i_step] = train_reward

            env.Compute_Interference(action_channel)
            # get new state
            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = env.get_state([i, j])
                    state_new_all.append((state_new))

            state_old_all = state_new_all

            memory.rewards.append(train_reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                time_step = 0

        Sum_rate_list.append(np.mean(Sum_rate_per_episode))
        Sum_AoI_list.append((np.mean(Sum_AoI_per_episode)))
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('step:', i_episode, 'reward', average_reward)
        print('Sum_rate_per_episode:', round(np.average(Sum_rate_per_episode), 2))
        print('Sum_AoI_per_episode:', round(np.average(Sum_AoI_per_episode), 2))

        if (i_episode+1) % 100 == 0 and i_episode != 0:
            agent.save_model(model_path)
    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    plt.figure(1)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    y2 = Sum_AoI_list
    plt.figure(2)
    plt.plot(x, y2)
    plt.xlabel('Episode')
    plt.ylabel('Sum AoI')
    y3 = Sum_rate_list
    plt.figure(3)
    plt.plot(x, y3)
    plt.xlabel('Episode')
    plt.ylabel('Sum rate')
    plt.show()

    print('Training Done. Saving models...')
    '''current_dir = os.path.dirname(os.path.realpath(__file__))

    reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')

    scipy.io.savemat(reward_path, {'reward': record_reward_average})'''

