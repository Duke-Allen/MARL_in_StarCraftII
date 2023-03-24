import time
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from agent.agent import Agents
from common.hams_rollout import HAMSRolloutWorker


class HAMSRunner:
    def __init__(self, env, args):
        # Env
        env.reset()  # 先渲染环境方便获取地图中智能体的各类信息
        env.close()
        self.env = env
        self.args = args
        self.envs = args.envs
        self.eval_envs = args.eval_envs
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.episode_length = args.episode_limit
        # self.episode_length = args.episode_length

        # Algorithm
        self.agents = Agents(env, args)
        self.rolloutWorker = HAMSRolloutWorker(self.agents, args)

        # The result for evaluation
        self.win_rates = []
        self.episode_rewards = []

        # Parameters
        self.num_env_steps = args.num_env_steps
        self.n_rollout_threads = args.n_rollout_threads
        self.n_eval_rollout_threads = args.n_eval_rollout_threads

        # Interval
        self.n_episodes = args.n_episodes
        self.eval_interval = args.evaluate_cycle  # evaluate interval

        # Save path for plt and pkl files
        self.save_path = args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Noise ---add
        self.noise_vector = None
        self.reset_noise()

    def reset_noise(self):
        # init noise
        if self.noise_vector is None:
            self.noise_vector = []
            for i in range(self.n_agents):
                self.noise_vector.append(np.random.randn(self.args.noise_dim) * self.args.sigma)
            self.noise_vector = np.array(self.noise_vector)
        else:
            # shuffle noise
            np.random.shuffle(self.noise_vector)

    def run(self, num):
        train_steps = 0
        # Total number of episodes
        episode_num = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episode_num = episode_num + 1 if episode_num % self.args.save_cycle == 0 else episode_num  # to save the last episode model
        print('Total episode number: {}'.format(episode_num))
        # for episode in tqdm(range(episode_num)):
        #     print('Run {}, train episode {}'.format(num, episode))
        for episode in tqdm(range(self.args.n_epoch)):
            print('Run {}, train epoch {}'.format(num, episode))
            # 每eval_interval个episode后进行一次计算胜率
            if episode % self.eval_interval == 0:
                win_rate, episode_reward = self.evaluate(self.noise_vector)  # return the win_rate and cumulative_reward
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)

            # reset noise ---add
            if self.args.use_value_noise and self.args.reset_interval != -1:
                if episode % self.args.reset_interval == 0:
                    print('shuffle noise Vector...')
                    self.reset_noise()

            # Collect episode samples
            episodes = []
            for episode_idx in range(self.n_episodes):
                sample, _ = self.rolloutWorker.generate_train_episode(self.noise_vector, episode_idx)
                episodes.append(sample)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_sample = episodes[0]
            episodes.pop(0)
            for e in episodes:
                for key in episode_sample.keys():
                    episode_sample[key] = np.concatenate((episode_sample[key], e[key]), axis=0)

            # Train
            self.agents.train(episode_sample, train_steps, num, self.noise_vector, self.rolloutWorker.epsilon)
            train_steps += 1
        # This Run is end, plot the picture after last epoch train
        win_rate, episode_reward = self.evaluate(self.noise_vector)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self, noise_vector):
        win_number = 0
        episode_rewards = 0
        for episode_id in range(self.args.evaluate_epoch):
            episode_reward, win_tag = self.rolloutWorker.generate_evaluate_episode(noise_vector, episode_num=episode_id, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
            # print("Returns {}, win_number {}".format(episode_reward, win_number))
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        # plt.axis([0, self.args.n_epoch, 0, 100])
        plt.ylim([0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.eval_interval))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.eval_interval))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()
