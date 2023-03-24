import numpy as np
import torch
from torch.distributions import Categorical
from common.utils import merge_batch


# Agent no communication
class Agents:
    def __init__(self, env, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        args.env = env
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            from policy.maven import MAVEN
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif args.alg == 'msmarl':  # MS-MARL算法
            from policy.msmarl import MSMARL
            self.policy = MSMARL(args)
        elif args.alg == 'msmarl+communication':  # ms-marl引入slave通信
            from policy.msmarl import MSMARL
            self.policy = MSMARL(args)
        elif args.alg == 'hams':  # HAMS算法
            from policy.hams import HAMS
            self.policy = HAMS(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init Agents')

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()  # (obs_shape,)
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.  # agent_num表示第几个智能体

        if self.args.last_action:  # last_action维度 (n_actions,)
            inputs = np.hstack((inputs, last_action))  # 将智能体自身观测与上一刻动作按水平方向拼接
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))  # 将inputs与智能体个数按水平方向拼接
        hidden_state = self.policy.eval_hidden[:, agent_num, :]  # (episode_num, rnn_hidden_dim)，将agent_num智能体的隐藏状态提出

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        # transform the shape of avail_actions from (n_actions,) to (1, n_actions)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action

    def master_choose_action(self, state, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        # (n_slave, n_actions) (n_slave, slave_hidden_dim)
        slave_self_a_prob, slave_hidden_state = self.get_slave_self_action_prob(obs, last_action)

        # 计算master输入
        slave_mean_hidden = torch.mean(slave_hidden_state, dim=0, keepdim=True).expand(self.policy.n_master, -1)  # (1, slave_hidden_dim)
        global_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).expand(self.policy.n_master, -1)  # (1, state_shape)
        master_input = list()
        master_input.append(global_state)
        master_input.append(slave_mean_hidden.cpu())
        master_input = torch.cat([x for x in master_input], dim=1)  # (n_master, state_shape + slave_hidden_dim)

        # transform the shape of avail_actions from (n_actions,) to (1, n_actions)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.args.cuda:
            master_input = master_input.cuda()
            self.policy.h_master = self.policy.h_master.cuda()
            self.policy.c_master = self.policy.c_master.cuda()
            slave_hidden_state = slave_hidden_state.cuda()

        # 输出参数维度第一个：(n_master, n_actions)，第二个：(n_slave, n_actions)，后两个：(n_master, master_hidden_dim)
        _, master_slave_a_prob, self.policy.h_master, self.policy.c_master = self.policy.master(master_input,
                                                                                                            self.policy.h_master,
                                                                                                            self.policy.c_master,
                                                                                                            slave_hidden_state,
                                                                                                            self.policy.n_slave)
        # # choose action from probability
        # action = self._choose_action_from_softmax(master_a_prob.cpu(), avail_actions, epsilon, evaluate)
        # return action, master_slave_a_prob, slave_self_a_prob
        action_prob = slave_self_a_prob + master_slave_a_prob  # all slave action probabilities: (n_agents, n_actions)
        return action_prob

    def slave_choose_action(self, all_action_prob, agent_num, avail_actions, epsilon, evaluate=False):
        # (n_slave, n_actions)
        # slave_action_prob = m2s_a_prob + s2s_a_prob
        inputs = all_action_prob[agent_num, :].unsqueeze(0)  # (1, n_actions)

        # transform the shape of avail_actions from (n_actions,) to (1, n_actions)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        # choose action from probability
        action = self._choose_action_from_softmax(inputs.cpu(), avail_actions, epsilon, evaluate)
        return action

    def get_slave_self_action_prob(self, obs, last_action):
        """
        计算所有slave自身输出的动作概率，输出除动作概率外还包含slave在t时刻的隐藏状态h
        obs为env.get_obs()得到，last_action为所有智能体上一刻动作的onehot编码
        输出为slave计算的自身动作概率，以及隐藏状态ht
        """
        slave_input = np.array(obs)  # 由列表变为npArray (n_agents, obs_shape)
        # if self.args.master_index == 0:
        #     slave_input = np.array([slave_obs for slave_obs in obs[1:]])  # 由列表变为npArray
        # else:
        #     slave_input = np.array([slave_obs for slave_obs in obs[:self.args.master_index]])  # 由列表变为npArray

        # 给inputs添加上一个动作、agent编号
        if self.args.last_action:
            slave_input = np.concatenate((slave_input, last_action), axis=1)  # 在第二个维度拼接加上n_actions
            # if self.args.master_index == 0:
            #     slave_input = np.concatenate((slave_input, last_action[1:, :]), axis=1)  # 在第二个维度拼接加上n_actions
            # else:
            #     slave_input = np.concatenate((slave_input, last_action[:self.args.master_index, :]), axis=1)  # 在第二个维度拼接加上n_actions
        if self.args.reuse_network:
            slave_input = np.concatenate((slave_input, np.eye(self.policy.n_slave)), axis=1)  # 在第二个维度拼接加上n_slave
        hidden_state = self.policy.eval_hidden  # (1, n_slave, slave_hidden_dim)
        hidden_state = hidden_state.reshape(1 * self.policy.n_slave, -1)  # (1 * n_slave, slave_hidden_dim)

        slave_input = torch.tensor(slave_input, dtype=torch.float32)  # transform npArray to tensor

        if self.args.cuda:  # create the tensor on the GPU device
            slave_input = slave_input.cuda()
            hidden_state = hidden_state.cuda()

        # 输出自身算出的动作概率，以及隐藏状态 (n_slave, n_actions)、(n_slave, slave_hidden_dim)
        a_self_prob, self.policy.eval_hidden = self.policy.slave(slave_input, hidden_state)
        return a_self_prob, self.policy.eval_hidden

    def hams_calculate_dist(self, state, obs, last_action, avail_actions, epsilon, noise_vector, evaluate=False):
        t_episode = merge_batch(state, obs, last_action, avail_actions)  # dim: (n_threads, n_agents, shape)
        for key in t_episode.keys():
            t_episode[key] = torch.tensor(t_episode[key], dtype=torch.float32)
        # 得到输出分布和对应此时的估值
        # dist: (batch_size * episode_len * n_agents, n_actions)
        # v_eval: (batch_size * episode_len * n_agents, 1)
        dist, v_eval = self.policy.model(t_episode, epsilon, noise_vector, evaluate)
        actions = dist.sample().long()  # 采样得到所有智能体的动作 (batch_size * n_agents,)
        actions_log_prob = dist.log_prob(actions)  # 所有智能体动作概率对数 (batch_size * n_agents,)

        # Transform shape to (n_threads, n_agents, 1)
        actions = actions.unsqueeze(-1).reshape(-1, self.n_agents, 1)
        actions_log_prob = actions_log_prob.unsqueeze(-1).reshape(-1, self.n_agents, 1)
        if not evaluate:
            v_eval = v_eval.reshape(-1, self.n_agents, 1)

        return actions, actions_log_prob, v_eval

    def hams_choose_action(self, all_actions, all_actions_log_prob, agent_num, epsilon, evaluate=False):
        # all_actions: 当前智能体动作(n_threads, n_agents, 1)
        # all_actions_log_prob: 当前智能体动作概率对数(n_threads, n_agents, 1)
        action = all_actions[:, agent_num].squeeze(-1)
        action_log_prob = all_actions_log_prob[:, agent_num].detach().squeeze(-1)

        return action, action_log_prob

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon 探索 ε：探索概率
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        # 选择动作基准
        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, run_num, noise_vector, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        if self.args.alg == 'hams':
            # max_episode_len = self.args.episode_length
            max_episode_len = self.args.episode_limit
        else:
            max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, noise_vector, epsilon)  # train the network parameters
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(run_num, train_step)


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            from policy.coma import COMA
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif alg.find('msmarl') > -1:
            from policy.msmarl import MSMARL
            self.policy = MSMARL(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        # 计算各智能体动作的权重，obs:(n_agents, obs_shape)，last_action:(n_agents, n_actions)
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, run_num, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(run_num, train_step)
