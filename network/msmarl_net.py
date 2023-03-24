import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import itertools

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class Master(nn.Module):
    def __init__(self, input_shape, args):
        super(Master, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.master_hidden_dim)
        # 使用了单层的LSTM(nn.LSTM不严谨地理解，可以认为是多层的LSTMCell，其输入为整个时间序列，LSTMCell则相当于在一个时间步上的处理)
        self.lstm = nn.LSTMCell(args.master_hidden_dim, args.master_hidden_dim)
        self.fc2 = nn.Linear(args.master_hidden_dim, args.n_actions)

        self.lstm1 = nn.LSTMCell(args.slave_hidden_dim, args.master_hidden_dim)
        self.fc3 = nn.Linear(args.master_hidden_dim, args.n_actions)  # 由master给出的：智能体各动作的概率

    def forward(self, obs, hidden_state, cell_state, h_slave, slave_num):
        # 注意：obs包括了master自身观测ot和slave隐藏状态平均mt，hidden_state为上一时刻隐藏状态，cell_state为上一时刻细胞状态
        # h_slave为t时刻的slave隐藏状态，用于结合t时刻的h c计算master给slave的动作指导概率
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.master_hidden_dim)
        c_in = cell_state.reshape(-1, self.args.master_hidden_dim)

        h, c = self.lstm(x, (h_in, c_in))
        output = self.fc2(h)  # output为master的动作概率

        # 原始h、c维度均为(n_master, master_hidden_dim)，与h_slave(n_slave, slave_hidden_dim)的batch大小不匹配，将其进行扩展
        h_gcm = h.repeat(slave_num, 1)  # GCM门控输入
        c_gcm = c.repeat(slave_num, 1)
        a_prob, _ = self.lstm1(h_slave, (h_gcm, c_gcm))
        a_prob_master = self.fc3(a_prob)  # master给slave指导的动作概率

        return output, a_prob_master, h, c


class Slave(nn.Module):
    def __init__(self, input_shape, args):
        super(Slave, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.slave_hidden_dim)
        # 此处同样使用了GRUCell，与LSTMCell类似
        self.gru = nn.GRUCell(args.slave_hidden_dim, args.slave_hidden_dim)
        self.fc2 = nn.Linear(args.slave_hidden_dim, args.n_actions)  # 自身计算的：单个智能体的动作概率

        # self.lstm = nn.LSTMCell(args.slave_hidden_dim, args.master_hidden_dim)
        # self.fc3 = nn.Linear(args.master_hidden_dim, args.n_actions)  # 由master给出的：智能体各动作的概率

    def forward(self, obs, hidden_state):
        """
        obs仅为slave自身观测，按照MS-MARL算法说明其输入仅为观测和上一时刻隐藏状态
        """
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.slave_hidden_dim)

        h = self.gru(x, h_in)
        a_prob_self = self.fc2(h)

        # a_prob, _ = self.lstm(h, (h_m, c_m))  # GCM门控
        # a_prob_master = self.fc3(a_prob)
        # # master的动作指导: a_prob_from_master
        # # slave自身观测计算出的动作概率: a_prob_from_self
        # # slave自身的隐藏状态: h
        # # slave_action_prob表示与master指导结合后的动作概率
        # slave_action_prob = a_prob_self + a_prob_master
        return a_prob_self, h


# Slave for communication with other agents observation
class CommSlave(nn.Module):
    def __init__(self, input_shape, args):
        super(CommSlave, self).__init__()
        self.args = args

        # Encoding
        self.encoding = nn.Linear(input_shape, args.slave_hidden_dim)  # encoding all agents observation
        self.gru = nn.GRUCell(args.slave_hidden_dim, args.slave_hidden_dim)  # each agent input encoding, output the hidden state, to record past observation

        # Attention, same as the soft attention in G2Anet, calculate the edge weight
        self.q = nn.Linear(args.slave_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.slave_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.slave_hidden_dim, args.attention_dim)

        # Decoding: input each agent's x_i and h_i, output agents actions probability distribution
        self.decoding = nn.Linear(args.slave_hidden_dim + args.attention_dim, args.n_actions)
        self.gnn = GCNConv(args.slave_hidden_dim + args.attention_dim, args.n_actions)
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):
        # Encoding the agent obs firstly, obs: (batch_size * n_slave, input_shape)
        obs_encoding = f.relu(self.encoding(obs))  # obs_encoding: (batch_size * n_slave, args.slave_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.slave_hidden_dim)  # h_in: (batch_size * n_slave, args.slave_hidden_dim)

        # Get the h_out through the agent GRU, h_out: (batch_size * n_slave, args.slave_hidden_dim)
        h_out = self.gru(obs_encoding, h_in)  # hidden state for slave agent

        # Soft attention
        q = self.q(h_out).reshape(-1, self.args.n_slave, self.args.attention_dim)  # (batch_size, n_slave, attention_dim)
        k = self.k(h_out).reshape(-1, self.args.n_slave, self.args.attention_dim)  # (batch_size, n_slave, attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.n_slave, self.args.attention_dim)  # (batch_size, n_slave, attention_dim)
        x = []  # Calculate the other agent's contribution for agent_i
        if self.args.n_slave > 1:
            # Number of slave agents is more than 1
            for i in range(self.args.n_slave):
                q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # agent_i q, (batch_size, 1, attention_dim)
                k_i = [k[:, j] for j in range(self.args.n_slave) if j != i]  # 对于agent_i来说其他agent的k，是n_slave-1个元素的列表，每个元素(batch_size, attention_dim)
                v_i = [v[:, j] for j in range(self.args.n_slave) if j != i]  # 对于agent_i来说其他agent的v，是n_slave-1个元素的列表，每个元素(batch_size, attention_dim)

                k_i = torch.stack(k_i, dim=0)  # (n_slave - 1, batch_size, attention_dim)
                k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, attention_dim, n_slave - 1)
                v_i = torch.stack(v_i, dim=0)  # (n_slave - 1, batch_size, attention_dim)
                v_i = v_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, attention_dim, n_slave - 1)
                # (batch_size, 1, attention_dim) * (batch_size, attention_dim, n_slave - 1) = (batch_size, 1, n_slave - 1)
                score = torch.matmul(q_i, k_i)

                # Normalization (batch_size, 1, n_slave - 1)
                scaled_score = score / np.sqrt(self.args.attention_dim)

                # Get the weights of other agent by softmax, (batch_size, 1, n_slave - 1)
                soft_weight = f.softmax(scaled_score, dim=-1)

                # 加权求和，矩阵的最后一维均是n_slave-1维度
                # (batch_size, attention_dim, n_slave - 1) * (batch_size, 1, n_slave - 1) = (batch_size, attention_dim, n_slave - 1)
                # 此处与原始代码一致，用直接求和代替GNN，在最后一个维度上求和得到x_i维度(batch_size, attention_dim)
                x_i = (v_i * soft_weight).sum(dim=-1)
                x.append(x_i)
        else:
            # Number of slave agents is 1. The attention mechanism is not used
            x.append(v[:, 0])

        # 得到的x是含有n_slave个元素的列表，列表每个元素的维度是(batch_size, attention_dim)，拼成一个tensor，维度(batch_size, n_slave, attention_dim)
        x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # reshape后(batch_size * n_slave, attention_dim)
        # 合并每个agent的h和x，final_output维度(batch_size * n_slave, slave_hidden_dim + attention_dim)
        final_output = torch.cat([h_out, x], dim=-1)

        # output is the action probabilities for slave agents
        output = self.decoding(final_output)  # (batch_size * n_slave, n_actions)

        return output, h_out

