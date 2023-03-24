import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import itertools
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from common.utils import check


# Master in one group for communication with other groups information，CommMaster是一个集群的master
class CommMaster(nn.Module):
    def __init__(self, input_shape, args):
        super(CommMaster, self).__init__()
        self.args = args

        # Attention
        # input: h_s'(batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)
        # output: each slave contribution in game, namely the slave weight for master (batch_size * episode_len * n_slave, master_hidden_dim)
        self.encoding = nn.Linear(input_shape, args.master_hidden_dim)

        self.q = nn.Linear(args.master_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.master_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.master_hidden_dim, args.attention_dim)

        self.decoding = nn.Linear(args.state_shape + args.attention_dim, args.master_hidden_dim)
        self.lstm = nn.LSTMCell(args.master_hidden_dim, args.master_hidden_dim)

    def forward(self, state, hidden_state, cell_state, h_slave, n_slaves):
        # episode_num = state.shape[0]  # batch size, namely episode_num
        # episode_len = state.shape[1]  # episode_len

        # state为master的全局状态(batch_size, episode_len, state_shape)，需要与经过Attention后的slave的h拼接送入LSTM，h_slave为t时刻的slave隐藏状态
        state = state.reshape(-1, self.args.state_shape)  # (batch_size * episode_len, state_shape)
        h_in = hidden_state.reshape(-1, self.args.master_hidden_dim)  # (batch_size * episode_len * 1, master_hidden_dim)
        c_in = cell_state.reshape(-1, self.args.master_hidden_dim)  # (batch_size * episode_len * 1, master_hidden_dim)

        # Encoding 从(batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)变为(batch_size * episode_len * n_slave, master_hidden_dim)
        input_encoding = f.relu(self.encoding(h_slave))

        # 将h_m维度变到attention_dim，方便进行attention操作 (batch_size * episode_len * 1, attention_dim)
        q = self.q(h_in)

        # Attention
        k = self.k(input_encoding).reshape(-1, n_slaves, self.args.attention_dim)  # (batch_size*episode_len, n_slave, attention_dim)
        v = f.relu(self.v(input_encoding)).reshape(-1, n_slaves, self.args.attention_dim)  # (batch_size*episode_len, n_slave, attention_dim)

        q_m = q.view(-1, 1, self.args.attention_dim)  # master的q (batch_size*episode_len, 1, attention_dim)
        # 对master来说，slave的k
        k_m = k.permute(0, 2, 1)  # 交换维度，(batch_size*episode_len, attention_dim, n_slave)
        # 对master来说，slave的v
        v_m = v.permute(0, 2, 1)  # 交换维度，(batch_size*episode_len, attention_dim, n_slave)
        # (batch_size*episode_len, 1, attention_dim) * (batch_size*episode_len, attention_dim, n_slave) = (batch_size*episode_len, 1, n_slave)
        score = torch.matmul(q_m, k_m)
        # 正则化 (batch_size*episode_len, 1, n_slave)
        scaled_score = score / np.sqrt(self.args.attention_dim)
        # 得到slave的权重 (batch_size*episode_len, 1, n_slave)
        slave_weight = f.softmax(scaled_score, dim=-1)
        # 加权求和，矩阵最后一维是n_slave
        # (batch_size * episode_len, attention_dim, n_slave) * (batch_size * episode_len, 1, n_slave) = (batch_size * episode_len, attention_dim, n_slave)
        # 此处与原始代码一致，用直接求和代替GNN，在最后一个维度上求和得到维度(batch_size * episode_len, attention_dim)
        i_m = (v_m * slave_weight).sum(dim=-1)

        # build inputs
        inputs = [state, i_m]

        # 将master自身全局状态与slave经过Attention后的h拼接起来 (batch_size * episode_len, state_shape + attention_dim)
        # input_m = torch.cat([x.reshape(episode_num * episode_len, -1) for x in inputs], dim=-1)
        input_m = torch.cat(inputs, dim=-1)
        # 对拼接信息编码
        input_encode = f.relu(self.decoding(input_m))  # (batch_size * episode_len, args.master_hidden_dim)
        h_out, c_out = self.lstm(input_encode, (h_in, c_in))  # h, c: (batch_size * episode_len * 1, master_hidden_dim)
        h_out = h_out.reshape(-1, 1, self.args.master_hidden_dim)  # (batch_size * episode_len, 1, master_hidden_dim)
        c_out = c_out.reshape(-1, 1, self.args.master_hidden_dim)

        return h_out, c_out


# Slave for communication with other agents observation，CommSlave是一个集群内的一个slave
class CommSlave(nn.Module):
    def __init__(self, input_shape, args):
        super(CommSlave, self).__init__()
        self.args = args

        # Encoding
        self.encoding = nn.Linear(input_shape, args.slave_hidden_dim)  # encoding all agents observation
        self.gru = nn.GRUCell(args.slave_hidden_dim, args.slave_hidden_dim)  # each agent input encoding, output the hidden state, to record past observation

        # Attention, same as the soft attention in G2ANet, calculate the edge weight
        self.q = nn.Linear(args.slave_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.slave_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.slave_hidden_dim, args.attention_dim)

        # Decoding: input each agent's x_i and h_i, output agents actions probability distribution
        self.decoding = nn.Linear(args.slave_hidden_dim + args.attention_dim, args.n_actions)
        self.gnn = GCNConv(args.slave_hidden_dim + args.attention_dim, args.n_actions)
        self.input_shape = input_shape

    def forward(self, obs, hidden_state, slave_num):
        # slave_num为当前集群的slave数量
        # Encoding the agent obs firstly, obs: (batch_size * episode_len * n_slave, input_shape)
        obs_encoding = f.relu(self.encoding(obs))  # obs_encoding: (batch_size * episode_len * n_slave, args.slave_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.slave_hidden_dim)  # h_in: (batch_size * episode_len * n_slave, args.slave_hidden_dim)

        # Get the h_out through the agent GRU, h_out: (batch_size * episode_len * n_slave, args.slave_hidden_dim)
        h_out = self.gru(obs_encoding, h_in)  # hidden state for slave agent

        # Soft attention
        q = self.q(h_out).reshape(-1, slave_num, self.args.attention_dim)  # (batch_size*episode_len, n_slave, attention_dim)
        k = self.k(h_out).reshape(-1, slave_num, self.args.attention_dim)  # (batch_size*episode_len, n_slave, attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, slave_num, self.args.attention_dim)  # (batch_size*episode_len, n_slave, attention_dim)
        x = []  # Calculate the other agent's contribution for agent_i
        if slave_num > 1:
            # Number of slave agents is more than 1
            for i in range(slave_num):
                q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # agent_i q, (batch_size*episode_len, 1, attention_dim)
                k_i = [k[:, j] for j in range(slave_num) if j != i]  # 对于agent_i来说其他agent的k，是n_slave-1个元素的列表，每个元素(batch_size*episode_len, attention_dim)
                v_i = [v[:, j] for j in range(slave_num) if j != i]  # 对于agent_i来说其他agent的v，是n_slave-1个元素的列表，每个元素(batch_size*episode_len, attention_dim)

                k_i = torch.stack(k_i, dim=0)  # (n_slave - 1, batch_size*episode_len, attention_dim)
                k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size*episode_len, attention_dim, n_slave - 1)
                v_i = torch.stack(v_i, dim=0)  # (n_slave - 1, batch_size*episode_len, attention_dim)
                v_i = v_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size*episode_len, attention_dim, n_slave - 1)
                # (batch_size*episode_len, 1, attention_dim) * (batch_size*episode_len, attention_dim, n_slave - 1) = (batch_size*episode_len, 1, n_slave - 1)
                score = torch.matmul(q_i, k_i)

                # Normalization (batch_size*episode_len, 1, n_slave - 1)
                scaled_score = score / np.sqrt(self.args.attention_dim)

                # Get the weights of other agent by softmax, (batch_size*episode_len, 1, n_slave - 1)
                soft_weight = f.softmax(scaled_score, dim=-1)

                # 加权求和，矩阵的最后一维均是n_slave-1维度
                # (batch_size*episode_len, attention_dim, n_slave - 1) * (batch_size*episode_len, 1, n_slave - 1) = (batch_size*episode_len, attention_dim, n_slave - 1)
                # 此处与原始代码一致，用直接求和代替GNN，在最后一个维度上求和得到x_i维度(batch_size*episode_len, attention_dim)
                x_i = (v_i * soft_weight).sum(dim=-1)
                x.append(x_i)
        else:
            # Number of slave agents is 1. The attention mechanism is not used
            x.append(v[:, 0])

        # 得到的x是含有n_slave个元素的列表，列表每个元素的维度是(batch_size*episode_len, attention_dim)，拼成一个tensor，维度(batch_size*episode_len, n_slave, attention_dim)
        x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # reshape后(batch_size * episode_len * n_slave, attention_dim)
        # 合并每个agent的h和x，new_h_out维度(batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)
        new_h_out = torch.cat([h_out, x], dim=-1)

        # output is the action probabilities for slave agents
        output = self.decoding(new_h_out)  # (batch_size * episode_len * n_slave, n_actions)智能体自身算出的策略

        return output, h_out, new_h_out


# 此处进行集群间Attention，再过LSTM得到Master的h_m和c_m，以便送给master进行指导，即此CommGroup为Attention以及LSTM编码
class CommGroup(nn.Module):
    def __init__(self, input_shape, args):
        super(CommGroup, self).__init__()
        self.args = args

        # Encoding
        # input_shape为(batch_size * episode_len * n_master, master_hidden_dim)
        # gcm_hidden_dim为模块内LSTM的隐藏状态维度，master的输出h_m'即(h_m, i_m)
        self.encoding = nn.Linear(input_shape, args.gcm_hidden_dim)  # 对各集群的信息[(h_m1, i_m1), (h_m2, i_m2), ...]进行编码

        # Soft attention
        self.q = nn.Linear(args.gcm_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.gcm_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.gcm_hidden_dim, args.attention_dim)

        # 第二层LSTM
        self.lstm = nn.LSTMCell(input_shape + args.attention_dim, args.gcm_hidden_dim)
        self.decoding = nn.Linear(args.gcm_hidden_dim, args.n_actions)  # 输出master自身的策略

    def forward(self, master_out, hidden_state, cell_state):
        # master_out为各集群的信息[(h_m1, i_m1), (h_m2, i_m2), ...]
        # master_out: (batch_size * episode_len * n_master, master_hidden_dim)
        # hidden_state: (batch_size * episode_len, n_master, gcm_hidden_dim)
        # cell_state: (batch_size * episode_len, n_master, gcm_hidden_dim)
        # 先对输入编码
        input_encoding = f.relu(self.encoding(master_out))  # (batch_size * episode_len * n_master, gcm_hidden_dim)

        # Soft Attention
        q = self.q(input_encoding).reshape(-1, self.args.n_master, self.args.attention_dim)  # (batch_size * episode_len, n_master, args.attention_dim)
        k = self.k(input_encoding).reshape(-1, self.args.n_master, self.args.attention_dim)  # (batch_size * episode_len, n_master, args.attention_dim)
        v = f.relu(self.v(input_encoding)).reshape(-1, self.args.n_master, self.args.attention_dim)  # (batch_size * episode_len, n_master, args.attention_dim)
        e = []  # 所有集群的信息
        if self.args.n_master > 1:
            # 有两个及以上的集群
            for i in range(self.args.n_master):
                q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # 集群i的q, (batch_size * episode_len, 1, args.attention_dim)
                k_i = [k[:, j] for j in range(self.args.n_master) if j != i]  # 对于集群i来说，其他集群的k
                v_i = [v[:, j] for j in range(self.args.n_master) if j != i]  # 对于集群i来说，其他集群的v

                k_i = torch.stack(k_i, dim=0)  # (n_master - 1, batch_size * episode_len, args.attention_dim)
                k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size * episode_len, args.attention_dim, n_master - 1)
                v_i = torch.stack(v_i, dim=0)  # (n_master - 1, batch_size * episode_len, args.attention_dim)
                v_i = v_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size * episode_len, args.attention_dim, n_master - 1)

                # (batch_size*episode_len, 1, attention_dim) * (batch_size*episode_len, attention_dim, n_master - 1) = (batch_size*episode_len, 1, n_master - 1)
                score = torch.matmul(q_i, k_i)

                # 归一化 (batch_size * episode_len, 1, n_master - 1)
                scaled_score = score / np.sqrt(self.args.attention_dim)

                # 经过softmax得到权重
                group_weights = f.softmax(scaled_score, dim=-1)  # (batch_size * episode_len, 1, n_master - 1)

                # 加权求和，矩阵的最后一维均是n_master-1维度
                # (batch_size*episode_len, attention_dim, n_master - 1) * (batch_size*episode_len, 1, n_master - 1) = (batch_size*episode_len, attention_dim, n_master - 1)
                # 此处与原始代码一致，用直接求和代替GNN，在最后一个维度上求和得到e_i维度(batch_size*episode_len, attention_dim)
                e_i = (v_i * group_weights).sum(dim=-1)
                e.append(e_i)
        else:
            # 只有一个集群，即所有智能体都是同构的，不需要Attention机制
            e.append(v[:, 0])  # (batch_size*episode_len, attention_dim)

        # 得到的e是含有n_master个元素的列表
        # 列表每个元素的维度是(batch_size*episode_len, attention_dim)，拼成一个tensor (batch_size*episode_len, n_master, attention_dim)
        e = torch.stack(e, dim=1).reshape(-1, self.args.attention_dim)  # reshape后(batch_size * episode_len * n_master, attention_dim)

        # 将输入的集群的h和i与集群的信息e合并，final_input维度(batch_size * episode_len * n_master, master_hidden_dim + attention_dim)
        final_input = torch.cat([master_out, e], dim=-1)
        h_in = hidden_state.reshape(-1, self.args.gcm_hidden_dim)  # (batch_size * episode_len * n_master, gcm_hidden_dim)
        c_in = cell_state.reshape(-1, self.args.gcm_hidden_dim)  # (batch_size * episode_len * n_master, gcm_hidden_dim)
        h_out, c_out = self.lstm(final_input, (h_in, c_in))  # 经过第二层LSTM得到每个集群master的新隐藏状态hm', cm'
        h_out = h_out.reshape(-1, self.args.n_master, self.args.gcm_hidden_dim)  # (batch_size * episode_len, n_master, gcm_hidden_dim)
        c_out = c_out.reshape(-1, self.args.n_master, self.args.gcm_hidden_dim)  # (batch_size * episode_len, n_master, gcm_hidden_dim)
        m_actions = self.decoding(h_out)  # 各集群master的策略 (batch_size * episode_len * n_master, n_actions)

        return m_actions, h_out, c_out


# 每个master都维护一个GCM，用以计算其对组内智能体的策略指导
class GCM(nn.Module):
    def __init__(self, input_shape, args):
        # input_shape为当前slave的隐藏状态h' (batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)
        super(GCM, self).__init__()
        self.args = args

        self.lstm = nn.LSTMCell(input_shape, args.gcm_hidden_dim)
        self.decoding = nn.Linear(args.gcm_hidden_dim, args.n_actions)

    def forward(self, h_m, c_m, h_slave, slave_num):
        # h_slave: (batch_size * episode_len * n_slave, slave_hidden_dim + attention_dim)
        # h_m与c_m维度均为(batch_size * episode_len, gcm_hidden_dim)，与h_slave的batch不匹配，将其扩展
        h_m = h_m.unsqueeze(1).expand(-1, slave_num, -1)  # (batch_size * episode_len, n_slave, gcm_hidden_dim)
        c_m = c_m.unsqueeze(1).expand(-1, slave_num, -1)
        h_gcm = h_m.reshape(-1, self.args.gcm_hidden_dim)  # (batch_size * episode_len * n_slave, gcm_hidden_dim)
        c_gcm = c_m.reshape(-1, self.args.gcm_hidden_dim)  # (batch_size * episode_len * n_slave, gcm_hidden_dim)

        h_out, c_out = self.lstm(h_slave, (h_gcm, c_gcm))  # 输出的h和c (batch_size * episode_len * n_slave, gcm_hidden_dim)
        output = self.decoding(h_out)  # master给slave的策略指导 (batch_size * episode_len * n_slave, n_actions)

        return output


# 对当前状态进行估计以作为baseline，输出为当前状态下的v值估计（各智能体一样）
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs, noise_vector):
        """
        输入：全局状态state: (batch_size, state_shape)
             noise_vector: (n_agents, noise_dim)
        输出：当前时刻的状态价值估计v: (batch_size, 1)
        """
        if self.args.use_value_noise:  # ---add
            N = self.args.n_agents
            noise_vector = check(noise_vector)
            if self.args.cuda:
                noise_vector = noise_vector.cuda()
            noise_vector = noise_vector.repeat(inputs.shape[0] // N, 1).float()
            inputs = torch.cat((inputs, noise_vector), dim=-1)

        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        v = self.fc3(x)
        return v
