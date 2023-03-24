import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean(method, map_name, num_run):
    path = []
    alg_num = 1
    win_rates = [[] for _ in range(alg_num)]
    win_rates_max = [[] for _ in range(alg_num)]
    win_rates_min = [[] for _ in range(alg_num)]
    game_map = map_name
    path.append('../result/{}/'.format(method) + game_map)
    # path.append('../result/qmix/' + game_map)
    # path.append('../result/qtran_base/' + game_map)
    # path.append('../result/qtran_alt/' + game_map)
    # path.append('../result/coma/' + game_map)
    # path.append('../result/central_v+commnet/' + game_map)
    # path.append('../result/central_v+g2anet/' + game_map)
    # path.append('../result/maven/' + game_map)
    # num_run = 8
    for i in range(alg_num):
        for j in range(num_run):
            win_rate_i = np.load(path[i] + '/win_rates_{}.npy'.format(j))
            win_rates[i].append(win_rate_i)

            # plt.plot(range(len(win_rate_i)), win_rate_i, c='#A0A0A0', linewidth=3)

    win_rates_max = np.max(win_rates[0], axis=0)
    win_rates_min = np.min(win_rates[0], axis=0)

    plt.plot(range(len(win_rates_max)), win_rates_max, c='#D0D0D0', linewidth=1)
    plt.plot(range(len(win_rates_max)), win_rates_min, c='#D0D0D0', linewidth=1)
    plt.fill_between(range(len(win_rates_max)), win_rates_min, win_rates_max, facecolor='#D0D0D0')
    win_rates = np.array(win_rates).mean(axis=1)
    new_win_rates = [[] for _ in range(alg_num)]
    average_cycle = 1
    for i in range(alg_num):
        rate = 0
        time = 0
        for j in range(len(win_rates[0])):
            rate += win_rates[i, j]
            time += 1
            if time % average_cycle == 0:
                new_win_rates[i].append(rate / average_cycle)
                time = 0
                rate = 0
    new_win_rates = np.array(new_win_rates)
    new_win_rates[:, 0] = 0
    win_rates = new_win_rates


    # plt.figure()
    plt.ylim(0, 1.0)
    plt.plot(range(len(win_rates[0])), win_rates[0], c='#000000', label='mean win rate')
    # plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='vdn')
    # plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='qmix')
    # plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qtran_base')
    # plt.plot(range(len(win_rates[3])), win_rates[3], c='y', label='qtran_alt')
    # plt.plot(range(len(win_rates[4])), win_rates[4], c='c', label='coma')
    # plt.plot(range(len(win_rates[7])), win_rates[7], c='#000000', label='maven')
    # plt.plot(range(len(win_rates[5])), win_rates[5], c='#FFA500', label='central_v+commnet')
    # plt.plot(range(len(win_rates[6])), win_rates[6], c='m', label='central_v+g2anet')

    plt.legend()
    plt.xlabel('episodes * 100')
    plt.ylabel('win_rate')
    plt.savefig('../result/overview_{}.png'.format(game_map))
    # plt.show()


def plt_reward_mean(method, map_name, num_run):
    path = []
    alg_num = 1
    rewards = [[] for _ in range(alg_num)]
    rewards_max = [[] for _ in range(alg_num)]
    rewards_min = [[] for _ in range(alg_num)]
    game_map = map_name
    path.append('../result/{}/'.format(method) + game_map)
    # path.append('../result/qmix/' + game_map)
    # path.append('../result/qtran_base/' + game_map)
    # path.append('../result/qtran_alt/' + game_map)
    # path.append('../result/coma/' + game_map)
    # path.append('../result/central_v+commnet/' + game_map)
    # path.append('../result/central_v+g2anet/' + game_map)
    # path.append('../result/maven/' + game_map)
    # num_run = 8
    for i in range(alg_num):
        for j in range(num_run):
            rewards_i = np.load(path[i] + '/episode_rewards_{}.npy'.format(j))
            rewards[i].append(rewards_i)

            # plt.plot(range(len(win_rate_i)), win_rate_i, c='#A0A0A0', linewidth=3)

    rewards_max = np.max(rewards[0], axis=0)
    rewards_min = np.min(rewards[0], axis=0)

    plt.plot(range(len(rewards_max)), rewards_max, c='#D0D0D0', linewidth=1)
    plt.plot(range(len(rewards_max)), rewards_min, c='#D0D0D0', linewidth=1)
    plt.fill_between(range(len(rewards_max)), rewards_min, rewards_max, facecolor='#D0D0D0')
    rewards = np.array(rewards).mean(axis=1)
    new_win_rates = [[] for _ in range(alg_num)]
    average_cycle = 1
    for i in range(alg_num):
        rate = 0
        time = 0
        for j in range(len(rewards[0])):
            rate += rewards[i, j]
            time += 1
            if time % average_cycle == 0:
                new_win_rates[i].append(rate / average_cycle)
                time = 0
                rate = 0
    new_win_rates = np.array(new_win_rates)
    new_win_rates[:, 0] = 0
    rewards = new_win_rates


    # plt.figure()
    plt.ylim(0, 20.0)
    plt.plot(range(len(rewards[0])), rewards[0], c='#000000', label='mean reward')
    # plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='vdn')
    # plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='qmix')
    # plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qtran_base')
    # plt.plot(range(len(win_rates[3])), win_rates[3], c='y', label='qtran_alt')
    # plt.plot(range(len(win_rates[4])), win_rates[4], c='c', label='coma')
    # plt.plot(range(len(win_rates[7])), win_rates[7], c='#000000', label='maven')
    # plt.plot(range(len(win_rates[5])), win_rates[5], c='#FFA500', label='central_v+commnet')
    # plt.plot(range(len(win_rates[6])), win_rates[6], c='m', label='central_v+g2anet')

    plt.legend()
    plt.xlabel('episodes * 100')
    plt.ylabel('win_rate')
    plt.savefig('../result/overview_{}.png'.format(game_map))
    # plt.show()


if __name__ == '__main__':
    map_name = "2s3z"
    method = "central_v+commnet"
    num_run = 8

    plt.subplot(2, 1, 1)
    plt_win_rate_mean(method, map_name, num_run)
    plt.subplot(2, 1, 2)
    plt_reward_mean(method, map_name, num_run)
    plt.show()
