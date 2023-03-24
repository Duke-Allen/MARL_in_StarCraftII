import numpy as np
import os
import torch
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm


class Runner:
    def __init__(self, env, args):
        env.reset()  # 先渲染环境方便获取地图中智能体的各类信息
        env.close()
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(env, args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if (args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1
                and args.alg.find('msmarl') == -1 and args.alg.find('hams') == -1):  # these algorithms are on-policy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0  # 用于控制更新target网络
        # print('Run {} start'.format(num))
        for epoch in tqdm(range(self.args.n_epoch)):
            print('Run {}, train epoch {}'.format(num, epoch))
            # 每evaluate_cycle个epoch后进行一次计算胜率
            if epoch % self.args.evaluate_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _ = self.rolloutWorker.generate_episode(episode_idx)  # return episode, episode_reward, win_tag
                episodes.append(episode)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 训练agent，每个epoch训练一次智能体
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1 \
                    or self.args.alg.find('msmarl') > -1 or self.args.alg.find('hams') > -1:
                self.agents.train(episode_batch, train_steps, num, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps, num, epsilon=None)
                    train_steps += 1
        # this Run is end, plot the picture after last epoch train
        win_rate, episode_reward = self.evaluate()
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):  # calculate the win number and episode_reward each evaluate_cycle epochs
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()
