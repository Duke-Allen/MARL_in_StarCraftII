import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from common.utils import compute_gae


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        # 垃圾命名方法的含义
        # o 观测
        # u 动作
        # u_prob 动作对应概率
        # r 即时奖励
        # s 全局状态
        # avail_u 可用动作
        # u_onehot 动作的onehot编码
        # terminated 终止序列
        # padded 填充序列
        o, u, u_prob, avail_u_prob, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], [], [], []
        values, log_probs, last_u_onehot = [], [], []
        # returns, advantage = [], []
        self.env.reset()
        terminated = False  # 终止标志
        win_tag = False  # 获胜标志
        step = 0
        episode_reward = 0  # cumulative rewards 累积奖励
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        last_u_onehot.append(last_action.copy())
        if self.args.alg.find('hams') == -1:
            self.agents.policy.init_hidden(1)
        else:
            self.agents.policy.model.init_hidden(1)

        # epsilon decay: episode or epoch, even step!
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            # 对3m来讲 前4位 各方向是否能移动
            # 3*5 = 15 位 敌人状态 每人5个 (available_to_attack, distance, relative_x, relative_y, health)
            # 2*5 = 10 位 队友状态 未开启last_move(visible, distance, relative_x, relative_y, health)
            # 最后一位 自身血量
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, probs, avail_probs, avail_actions, actions_onehot = [], [], [], [], []
            action_log_probs = []
            value = None
            if self.args.alg.find('msmarl') > -1:  # ms-marl系列算法采样过程不同
                ms_a_prob, ss_a_prob = None, None
                action_prob = None
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    if agent_id == 0:  # 第一个智能体当作master，其余智能体认为是slave，master输入包含state、从智能体ht平均
                        # action, ms_a_prob, ss_a_prob = self.agents.master_choose_action(state, obs, last_action, agent_id, avail_action, epsilon, evaluate)
                        action_prob = self.agents.master_choose_action(state, obs, last_action, agent_id, avail_action, epsilon, evaluate)
                    # else:
                    #     action = self.agents.slave_choose_action(ms_a_prob, ss_a_prob, agent_id, avail_action, epsilon, evaluate)
                    action = self.agents.slave_choose_action(action_prob, agent_id, avail_action, epsilon, evaluate)
                    # generate onehot vector of the action
                    action_onehot = np.zeros(self.n_actions)
                    action_onehot[action] = 1
                    actions.append(np.int(action))  # actions为各智能体在t时刻的动作
                    actions_onehot.append(action_onehot)  # actions_onehot为各智能体在t时刻的动作onehot编码
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot  # 更新agent_id智能体上一个动作onehot编码
            elif self.args.alg.find('hams') > -1:  # hams算法
                all_action, a_log_prob = None, None  # 所有智能体的动作、动作概率对数
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)  # 获取当前智能体的可选动作
                    if agent_id == 0:  # 第一个master直接算出所有智能体策略
                        all_avail_action = np.array([self.env.get_avail_agent_actions(x) for x in range(self.n_agents)])  # (n_agents, n_actions)
                        all_action, a_log_prob, value = self.agents.hams_calculate_dist(state, obs, last_action, all_avail_action, epsilon, evaluate)
                    action, action_log_prob = self.agents.hams_choose_action(all_action, a_log_prob, agent_id, epsilon, evaluate)
                    # generate onehot vector of the action
                    action_onehot = np.zeros(self.n_actions)
                    action_onehot[action] = 1
                    actions.append(np.int(action))  # actions为各智能体在t时刻的动作
                    action_log_probs.append(action_log_prob.numpy())  # 智能体t时刻动作概率对数
                    # probs.append(a_prob.detach().numpy().flatten())  # a_prob为智能体t时刻动作对应的概率
                    # avail_probs.append(avail_a_prob.detach().numpy().flatten())  # avail_a_prob为智能体t时刻可用动作的概率（正则化后）
                    actions_onehot.append(action_onehot)  # actions_onehot为各智能体在t时刻的动作onehot编码
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot  # 更新agent_id智能体上一个动作onehot编码
            else:
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    # if self.args.alg == 'maven':
                    #     action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                    #                                        avail_action, epsilon, maven_z, evaluate)
                    # else:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, None, evaluate)
                    # generate onehot vector of the action
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1  # [0, 0, 0, 1, 0, 0, 0, 0, 0]
                    actions.append(np.int(action))  # [3, 6, 9]
                    actions_onehot.append(action_onehot)
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            if self.args.alg.find('hams') > -1:  #
                # u_prob.append(probs)
                # avail_u_prob.append(avail_probs)
                values.append(value.cpu().detach().numpy())
                log_probs.append(np.reshape(action_log_probs, [self.n_agents, 1]))
                last_u_onehot.append(last_action.copy())
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # final obs
        obs = self.env.get_obs()  # add the terminated observation and state
        state = self.env.get_state()
        next_value = None
        if self.args.alg.find('hams') > -1:
            tensor_state = torch.tensor(state).unsqueeze(0)
            if self.args.cuda:
                tensor_state = tensor_state.cuda()
            next_value = self.agents.policy.eval_critic(tensor_state).repeat(self.n_agents, 1)
            next_value = next_value.cpu().detach().numpy()
            # values.append(next_value.detach().numpy())
        o.append(obs)
        s.append(state)
        o_next = o[1:]  # the next is training for target q network, for absolute policy-based method is needless
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for terminated obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        last_u_onehot = last_u_onehot[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):  # 对于正常结束的回合（获胜或失败）将step扩充至episode_limit保证各episode长度一致
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            if self.args.alg.find('hams') > -1:
                # u_prob.append(np.zeros([self.n_agents, 1]))
                # avail_u_prob.append(np.zeros([self.n_agents, self.n_actions]))
                values.append(np.zeros((self.n_agents, 1)))
                log_probs.append(np.zeros((self.n_agents, 1)))
                last_u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                # returns.append(np.zeros((self.n_agents, 1)))
                # advantage.append(np.zeros((self.n_agents, 1)))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        # # 计算GAE Advantage
        # if self.args.alg.find('hams') > -1:
        #     returns = compute_gae(next_value, values, r, padded, self.args)
        #     advantage = (np.array(returns) - np.array(values)).tolist()

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )

        if self.args.alg.find('hams') > -1:  # hams添加额外的样本信息
            # 计算GAE Advantage
            returns = compute_gae(next_value, values, r, padded, self.args)
            advantage = (np.array(returns) - np.array(values)).tolist()
            # episode['u_prob'] = u_prob.copy()
            # episode['avail_u_prob'] = avail_u_prob.copy()
            episode['log_probs'] = log_probs.copy()
            # last_u_onehot = last_u_onehot[:-1]  # 最后一个舍弃
            episode['last_u_onehot'] = last_u_onehot.copy()
            episode['values'] = values.copy()
            episode['returns'] = returns.copy()
            episode['advantages'] = advantage.copy()

        # add episode dim
        for key in episode.keys():
            if episode[key]:  # 不为空则处理，否则不处理
                episode[key] = np.array([episode[key]])
            # if key != 'u_prob' and key != 'avail_u_prob' and key != 'values':
            #     episode[key] = torch.from_numpy(np.array([episode[key]]))
            # elif episode[key]:  # 不为空则处理，否则不处理
            #     episode[key] = torch.stack(episode[key], dim=0).unsqueeze(0).detach()  # 在第一个维度上扩展出batch维度
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)

                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # add the final observation and state
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for final obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit, namely the battle win or lose. padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag
