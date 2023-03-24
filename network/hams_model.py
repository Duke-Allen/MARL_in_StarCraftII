import torch
import torch.nn as nn
from torch.distributions import Categorical


# 将算法整个前向过程合成一个大网络
class HamsModel(nn.Module):
    def __init__(self, args, slaves, masters, gcms, commgroup, critic):
        super(HamsModel, self).__init__()
        self.n_slaves = args.n_slaves
        self.n_master = args.n_master
        self.args = args

        #  cluster network
        self.slaves = slaves
        # self.masters = masters
        # self.commGroup = commgroup
        # self.gcms = gcms
        self.critic = critic

        # hidden state
        self.eval_hidden = []  # 为每个集群的slave维护一个eval_hidden，故是一个列表
        # self.h_master = []  # 同self.eval_hidden，master为LSTM故需要h和c
        # self.c_master = []
        # self.h_commGroup = None
        # self.c_commGroup = None

    def forward(self, batch, epsilon, noise_vector, evaluate=False):
        """
        计算所有智能体动作概率和估计v值。
        输入：
        batch：采样得到的全部样本：(batch_size, n_agents, shape)
        noise_vector: (n_agents, noise_dim)
        step：当前的时间步

        输出：
        distribution: (batch_size * episode_len * n_agents, n_actions)
        v_eval: (batch_size * episode_len, 1)
        """
        batch_size = batch["o"].shape[0]
        self.init_hidden(batch_size)
        # episode_num = batch["o"].shape[0]  # batch_size，即batch中episode数量
        # episode_len = batch["o"].shape[1]  # 样本的长度，可能为一个timestep或几个timestep
        # dim: (batch_size * episode_len, n_agent, shape)
        state, obs, avail_actions, last_u_onehot = \
            batch["s"], batch["o"], batch["avail_u"], batch["last_u_onehot"]
        # ***********************Policy Network***********************
        # SLAVE:
        # 给obs加last_action、agent_id，获取各集群slave的输入，列表元素维度(batch_size * episode_len * n_slave, input_shape)
        slave_inputs = []
        index = 0
        for i in range(len(self.n_slaves)):
            n_group_i = self.n_slaves[i]
            index += n_group_i
            obs_inputs_i = obs[:, index - n_group_i:index]
            last_u_onehot_inputs_i = last_u_onehot[:, index - n_group_i:index]
            slave_input_i = [obs_inputs_i]  # slave input: obs
            if self.args.last_action:
                slave_input_i.append(last_u_onehot_inputs_i)  # slave input: last action(onehot)
            if self.args.reuse_network:  # slave input: serial number
                '''
                因为当前的inputs三维的数据，每一维分别代表(episode编号, agent编号, inputs维度)，直接在dim=1上添加对应的向量即可
                比如给agent_0后面加(1, 0, 0, 0, 0), 表示5个agent中的0号，而agent_0的数据正好在第0行，那么需要加的agent编号恰好
                就是一个单位阵，即对角线为1，其余为0
                '''
                slave_input_i.append(torch.eye(self.n_slaves[i]).unsqueeze(0).expand(batch_size, -1, -1))
            # 要把slave的输入各部分拼起来，因为群内slave共享一个网络，每条数据中带上了自己的编号，所以还是自己的数据
            slave_input_i = torch.cat([x.reshape(batch_size * self.n_slaves[i], -1) for x in slave_input_i], dim=1)
            slave_inputs.append(slave_input_i)
        if self.args.cuda:
            for i in range(self.n_master):
                slave_inputs[i] = slave_inputs[i].cuda()
                self.eval_hidden[i] = self.eval_hidden[i].cuda()
                # self.h_master[i] = self.h_master[i].cuda()
                # self.c_master[i] = self.c_master[i].cuda()
            # self.h_commGroup = self.h_commGroup.cuda()
            # self.c_commGroup = self.c_commGroup.cuda()
        # 将slave自身观测、隐藏状态输入得到slave输出和当前时刻的隐藏状态
        # 输出第一个为智能体自身策略，第二个为h_slave，第三个为加入其他信息的h_slave'
        action_prob_ss, new_h_slave = [], []
        for i in range(len(self.n_slaves)):
            # action_prob_i: (batch_size * episode_len * n_slave, n_actions)
            # self.eval_hidden[i]: (batch_size * episode_len * n_slave, slave_hidden_dim)
            # new_h_slave_i: (batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)
            action_prob_i, self.eval_hidden[i], new_h_slave_i = self.slaves[i](slave_inputs[i],
                                                                               self.eval_hidden[i],
                                                                               self.n_slaves[i])
            # 变换动作概率维度并保存 (batch_size * episode_len, n_slave, n_actions)
            action_prob_ss.append(action_prob_i.view(-1, self.n_slaves[i], self.args.n_actions))
            new_h_slave.append(new_h_slave_i)

        # # MASTER:
        # # master_states维度(batch_size, episode_len, n_master, state_shape)
        # # new_h_slave为列表每个元素维度(batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)
        # # batch中s为(batch_size, episode_len, state_shape)
        # # s没有n_master维度，其他数据都是四维不能拼接，所以要把s转化为四维: (batch_size, episode_len, n_master, state_shape)
        # # master_states = state.unsqueeze(2).expand(-1, -1, self.n_master, -1)
        # master_states = state  # (episode_num * episode_len, n_agents, state_shape)
        # if self.args.cuda:
        #     master_states = master_states.cuda()
        #     # for i in range(len(new_h_slave)):
        #     #     new_h_slave[i] = new_h_slave[i].cuda()
        # for i in range(self.n_master):
        #     # self.h_master[i]: (batch_size * episode_len, 1, master_hidden_dim)
        #     # self.c_master[i]: (batch_size * episode_len, 1, master_hidden_dim)
        #     # 由全局状态和h_s得到master的h_m和c_m，后续还要经过commGroup、GCM两部分才能得到master的动作策略和其对slave的指导策略
        #     self.h_master[i], self.c_master[i] = self.masters[i](master_states[:, i],
        #                                                          self.h_master[i],
        #                                                          self.c_master[i],
        #                                                          new_h_slave[i],
        #                                                          self.n_slaves[i])
        #
        # # COMMGROUP:
        # # 将h_m进行拼接(batch_size * episode_len, n_master, master_hidden_dim)
        # comm_inputs = torch.cat(self.h_master, dim=-2)
        # comm_inputs = comm_inputs.reshape(-1, self.args.master_hidden_dim)  # (batch_size * episode_len * n_master, master_hidden_dim)
        # # h_comm, c_comm: (batch_size * episode_len, n_master, gcm_hidden_dim)
        # _, self.h_commGroup, self.c_commGroup = self.commGroup(comm_inputs, self.h_commGroup, self.c_commGroup)
        #
        # # GCM:
        # # 每个master的h_m'和c_m'送入GCM计算对组内slave的指导动作策略
        # action_prob_ms = []
        # for i in range(self.n_master):
        #     # action_prob_j: (batch_size * episode_len * n_slave, n_actions)
        #     action_prob_j = self.gcms[i](self.h_commGroup[:, i],
        #                                  self.c_commGroup[:, i],
        #                                  new_h_slave[i],
        #                                  self.n_slaves[i])
        #     action_prob_ms.append(action_prob_j.view(-1, self.n_slaves[i], self.args.n_actions))

        # Merge the slave policy and master policy
        s_actions = []
        for i in range(len(self.n_slaves)):
            # element dim: (batch_size * episode_len, n_slave, n_actions)
            s_actions.append(action_prob_ss[i])

        # 将各集群智能体动作策略拼接起来(batch_size * episode_len, n_agents, n_actions)
        output = torch.cat(s_actions, dim=-2)
        action_prob = torch.nn.functional.softmax(output, dim=-1).cpu().clone()  # 输出经过softmax归一化
        # action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, avail_actions.shape[-1])
        # action_prob = (1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num
        # print("After softmax:")
        # print(action_prob)
        # action_prob = torch.exp(output)
        # action_prob = torch.nn.functional.normalize(action_prob, p=1, dim=-1).cpu()

        # ***********************Value Network***********************
        if not evaluate:
            critic_inputs = state.reshape(-1, self.args.state_shape)
            if self.args.cuda:
                critic_inputs = critic_inputs.cuda()
            # input dim: (batch_size * episode_len * n_agents, state_shape), out dim: (batch_size * episode_len * n_agents, 1)
            v_eval = self.critic(critic_inputs, noise_vector)
        else:
            v_eval = None

        # avail_actions = avail_actions.reshape(-1, self.args.n_agents, self.args.n_actions)  # (batch_size * episode_len, n_agents, n_actions)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作置0
        action_prob = action_prob.reshape(-1, self.args.n_actions)  # (batch_size * episode_len * n_agents, n_actions)
        distribution = Categorical(action_prob)  # (batch_size * episode_len * n_agents, n_actions)
        if self.args.cuda and not evaluate:
            v_eval = v_eval.cuda()

        # if not evaluate:
        #     print("Final action_prob:")
        #     print(action_prob)

        return distribution, v_eval

    def ppo_generator(self, batch_data, num_mini_batch):
        """
        Yield num_mini_batch training datas for policy.
        输入：
        batch_data：采样得到的全部样本维度是(episode_num, episode_len, n_agents, shape)
        num_mini_batch：number of minibatch to split the batch into.

        输出：更新所用数据
        """
        episode_num, episode_len = batch_data["o"].shape[0], batch_data["o"].shape[1]
        batch_size = episode_num * episode_len  # 得到各集群的batch size
        mini_batch_sizes = batch_size // num_mini_batch  # number of samples in each minibatch

        rand = torch.randperm(batch_size).numpy()  # 抽样的mini batch数据索引
        sampler = [rand[i * mini_batch_sizes:(i + 1) * mini_batch_sizes] for i in range(num_mini_batch)]

        state = batch_data["s"].unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)  # (episode_num, episode_len, n_agents, state_shape)
        state = state.reshape(-1, self.args.n_agents, self.args.state_shape)  # (episode_num * episode_len, n_agents, state_shape)
        obs = batch_data["o"].reshape(-1, self.args.n_agents, self.args.obs_shape)  # (episode_num * episode_len, n_agents, obs_shape)
        available_actions = batch_data["avail_u"].reshape(-1, self.args.n_agents, self.args.n_actions)  # (episode_num * episode_len, n_agents, n_actions)
        last_actions = batch_data["last_u_onehot"].reshape(-1, self.args.n_agents, self.args.n_actions)  # (episode_num * episode_len, n_agents, n_actions)
        actions = batch_data["u"].reshape(-1, self.args.n_agents, 1)  # (episode_num * episode_len, n_agents, 1)
        value_preds = batch_data["values"].reshape(-1, self.args.n_agents, 1)  # (episode_num * episode_len, n_agents, 1)
        returns = batch_data["returns"].reshape(-1, self.args.n_agents, 1)  # (episode_num * episode_len, n_agents, 1)
        advantages = batch_data["advantages"].reshape(-1, self.args.n_agents, 1)  # (episode_num * episode_len, n_agents, 1)
        action_log_probs = batch_data["log_probs"].reshape(-1, self.args.n_agents, 1)  # (episode_num * episode_len, n_agents, 1)
        padded = batch_data["padded"].unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)  # (episode_num, episode_len, n_agents, 1)
        padded = padded.reshape(-1, self.args.n_agents, 1)  # (episode_num * episode_len, n_agents, 1)

        for indices in sampler:
            # size: (episode_num * episode_len, n_agents, shape)-->(mini_batch_size, n_agents, shape)
            state_batch = state[indices]
            obs_batch = obs[indices]
            available_actions_batch = available_actions[indices]
            last_actions_batch = last_actions[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            returns_batch = returns[indices]
            advantages_batch = advantages[indices]
            action_log_probs_batch = action_log_probs[indices]
            padded_batch = padded[indices]
            mini_batch_data = dict()
            mini_batch_data["s"] = state_batch
            mini_batch_data["o"] = obs_batch
            mini_batch_data["avail_u"] = available_actions_batch
            mini_batch_data["last_u_onehot"] = last_actions_batch
            mini_batch_data["u"] = actions_batch
            mini_batch_data["values"] = value_preds_batch
            mini_batch_data["returns"] = returns_batch
            mini_batch_data["advantages"] = advantages_batch
            mini_batch_data["log_probs"] = action_log_probs_batch
            mini_batch_data["padded"] = padded_batch
            yield mini_batch_data

    def init_hidden(self, batch_size):  # episode_num, episode_len
        """
        为所有网络初始化隐藏状态变量（使用的RNN结构）
        """
        # 初始化变量前先清空
        self.eval_hidden.clear()
        # self.h_master.clear()
        # self.c_master.clear()

        for i in range(self.n_master):
            self.eval_hidden.append(torch.zeros((batch_size, self.n_slaves[i], self.args.slave_hidden_dim)))
            # self.h_master.append(torch.zeros((batch_size, 1, self.args.master_hidden_dim)))
            # self.c_master.append(torch.zeros((batch_size, 1, self.args.master_hidden_dim)))
        # self.h_commGroup = torch.zeros((batch_size, self.n_master, self.args.gcm_hidden_dim))
        # self.c_commGroup = torch.zeros((batch_size, self.n_master, self.args.gcm_hidden_dim))

        # for i in range(self.n_master):
        #     self.eval_hidden.append(torch.zeros((batch_size, self.args.slave_hidden_dim)))
        #     self.h_master.append(torch.zeros((batch_size, self.args.master_hidden_dim)))
        #     self.c_master.append(torch.zeros((batch_size, self.args.master_hidden_dim)))
        # self.h_commGroup = torch.zeros((batch_size, self.args.gcm_hidden_dim))
        # self.c_commGroup = torch.zeros((batch_size, self.args.gcm_hidden_dim))
