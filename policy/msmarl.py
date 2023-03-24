import torch
import os
from network.base_net import RNN
from network.msmarl_net import *
from network.commnet import CommNet
from network.g2anet import G2ANet


class MSMARL:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.n_master = 1  # master数量（一个组）
        # self.n_slave = self.n_agents - self.n_master  # slave的数量（一个组）
        self.n_slave = self.n_agents  # number of slaves
        args.n_master = self.n_master
        args.n_slave = self.n_slave
        self.args = args

        master_input_shape = self._get_master_input_shape()  # master网络的输入维度
        slave_input_shape = self.obs_shape  # slave网络的输入维度和coma、vdn、qmix的rnn输入维度一样，使用同一个网络结构
        # 根据参数决定slave的输入维度
        if args.last_action:
            slave_input_shape += self.n_actions
        if args.reuse_network:
            slave_input_shape += self.n_slave  # slave共用一个网络

        # slave的神经网络，参数共享
        # 根据观测输出为下一时刻隐藏状态ht及智能体各动作概率，用概率选的时候还需要用softmax再运算一次
        if self.args.alg == "msmarl":
            print("Init alg msmarl")
            self.slave = Slave(slave_input_shape, args)
        # elif self.args.alg == "msmarl+commnet":  # 后期便于加通信，暂时不可用 21-11-23
        #     print("Init alg msmarl+commnet")
        #     self.slave = CommNet(slave_input_shape, args)
        # elif self.args.alg == "msmarl+g2anet":  # 后期便于加通信，暂时不可用 21-11-23
        #     print("Init alg msmarl+g2anet")
        #     self.slave = G2ANet(slave_input_shape, args)
        elif self.args.alg == "msmarl+communication":  # ms-marl算法的slave间进行attention通信
            print("Init alg msmarl+communication")
            self.slave = CommSlave(slave_input_shape, args)
        else:
            raise Exception("No such algorithm")

        # master的神经网络，参数共享
        self.master = Master(master_input_shape, self.args)

        # 是否使用gpu，是则放置在gpu上
        if self.args.cuda:
            self.slave.cuda()
            self.master.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + "/rnn_params.pkl"):
                path_rnn = self.model_dir + "/rnn_params.pkl"
                path_master = self.model_dir + "/master_params.pkl"
                map_location = "cuda:0" if self.args.cuda else "cpu"
                self.slave.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.master.load_state_dict(torch.load(path_master, map_location=map_location))
                print("Successfully load the model: {} and {}".format(path_rnn, path_master))
            else:
                raise Exception("No model!")

        # 对于ms-marl算法：
        # slave_parameters：0、1为fc1参数，其余部分为GRU参数
        # master_parameters：0、1为fc1参数，其余部分为LSTM参数
        self.slave_parameters = list(self.slave.parameters())
        self.master_parameters = list(self.master.parameters())
        if args.optimizer == "RMS":
            self.slave_optimizer = torch.optim.RMSprop(self.slave_parameters, lr=args.lr_slave)
            self.master_optimizer = torch.optim.RMSprop(self.master_parameters, lr=args.lr_master)
        self.args = args

        # 执行过程中，要为每个slave都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个slave都维护一个eval_hidden
        self.eval_hidden = None
        # master结构为LSTM，故需要h和c
        self.h_master = None
        self.c_master = None

        # master输出的给slave智能体的动作概率
        self.master_action_guide = None
        # 各slave智能体在t时刻的平均隐藏状态，作为master的输入
        self.slave_hidden_state = None

    # 获取master的输入维度
    def _get_master_input_shape(self):
        # master global state shape
        input_shape = self.state_shape
        # slave hidden state shape
        input_shape += self.args.slave_hidden_dim  # 算法中master输入包括slave隐藏状态的平均值
        # # agent_id
        # input_shape += self.n_agents

        return input_shape

    def learn(self, batch, max_episode_len, train_step, epsilon):
        """
        batch中每个数据都是四维，第一个维度为batch_size表示batch中有几个episode，第二个维度表示batch中的episode长度，
        第三个维度表示slave智能体数量，第四个维度表示具体的数据的维度（obs,state,action等）
        """
        episode_num = batch['o'].shape[0]  # batch中的episode个数
        self.init_hidden(episode_num)  # 为master和slave初始化隐藏状态
        # self.init_master_hidden(episode_num)
        for key in batch.keys():  # 把batch中的数据转为tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'], batch["avail_u"], batch["terminated"]
        # mask维度：(episode_num, max_episode_len, 1)，与batch['padded']维度一致，mask为1，padded为0
        mask = (1 - batch["padded"].float())  # 用来把那些填充的经验置0，从而不让他们影响到学习
        # master_mask = 1 - batch["padded"].float()  # 维度与mask一致，用于计算master时使用
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            mask = mask.cuda()
            # master_mask = master_mask.cuda()
            terminated = terminated.cuda()  # (episode_num, max_episode_len, 1)，终止step为1，其余为0

        # 得到每条经验的return，(episode_num, max_episode_len, n_agents)
        values = self._get_values(r, mask, terminated, max_episode_len)

        # 每个智能体的所有动作的概率, (episode_num, max_episode_len, n_agents, n_actions)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        # mask维度三维，repeat函数在第一个维度上复制为1个，在第二个维度上复制为1个，在第三个维度上复制为智能体数量
        # 给mask转换出n_agents维度，用于每个agent的训练
        mask = mask.repeat(1, 1, self.n_agents)
        # # 给master_mask转换出n_master维度，用于master的训练
        # master_mask = master_mask.unsqueeze(2).repeat(1, 1, self.n_master, self.args.master_hidden_dim)

        # dim表示在input上索引数据在第几维上，gather后的数据shape和index的shape一致
        # 如dim=3，input为四维(x1, x2, x3, x4)，则需要查找的数据索引形如(*, *, *, y)，y即为index中数值，而其余维度(*)索引则按index中位置排列
        # 取出智能体实际采取的动作的概率 (episode_num, max_episode_len, n_agents)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验概率为0，取log则为负无穷，故赋1取对数后为0
        log_pi_taken = torch.log(pi_taken)

        # 策略梯度方法的更新slave智能体参数
        # 将总loss分到每个episode中每个step时每个智能体上
        slave_loss = -((values * log_pi_taken) * mask).sum() / mask.sum()
        self.slave_optimizer.zero_grad()
        self.master_optimizer.zero_grad()
        slave_loss.backward()
        if self.args.alg == "reinforce+g2anet":
            pass
        self.slave_optimizer.step()
        self.master_optimizer.step()

        # # 更新master参数，用联合动作作为master的动作进行策略梯度的反向传播，更新方法待定
        # master_values = values.sum(dim=-1, keepdim=True)  # 将最后一维n_slave求和变为1，(episode_num, max_episode_len, 1)
        # # master_values扩展后维度：(episode_num, max_episode_len, n_master, master_hidden_dim)
        # # 即认为values在每个master是一样的
        # master_values = master_values.unsqueeze(2).expand(-1, -1, self.n_master, self.args.master_hidden_dim)
        # master_actions = batch["m_actions"]  # master的动作：(episode_num, max_episode_len, n_master, master_hidden_dim)
        # master_loss = -((master_values * master_actions) * master_mask).sum() / master_mask.sum()
        # self.master_optimizer.zero_grad()
        # master_loss.backward()
        # self.master_optimizer.step()

    def _get_master_inputs(self, batch, transition_idx, max_episode_len, slave_hidden_state):
        # transition_idx为一个episode中的第几个step，max_episode_len为batch中episode长度最大值
        # 目前认为第一个智能体为master，master有全局状态不需要自身观测，要取其他slave智能体的观测故从1开始
        obs, obs_next, s, s_next = batch["o"][:, transition_idx], batch["o_next"][:, transition_idx],\
                                   batch["s"][:, transition_idx], batch["s_next"][:, transition_idx]
        # if self.args.master_index == 0:
        #     obs, obs_next = batch["o"][:, transition_idx, 1:], batch["o_next"][:, transition_idx, 1:]
        # else:
        #     # the last agent is the master, obs of the slave agents must start from 0 to n_agents - 2
        #     obs, obs_next = batch["o"][:, transition_idx, :self.args.master_index], batch["o_next"][:, transition_idx, :self.args.master_index]
        # s, s_next = batch["s"][:, transition_idx], batch["s_next"][:, transition_idx]

        # batch中s为 (episode_num, max_episode_len, state_shape)，此函数中的s为从第二维度取出某一个时间步后，故不是三维
        # s是二维，没有n_agents维度，因为所有agent的s一样，其他数据都是三维不能拼接，所以要把s转化为三维：(episode_num, n_master, s_dim)
        s = s.unsqueeze(1).expand(-1, self.n_master, -1)
        episode_num = obs.shape[0]

        # # master除了全局状态外，还需要各slave智能体观测的平均值，obs取出后维度 (episode_num, n_slave, obs_dim)
        # # slave观测平均后：(episode_num, 1, obs_dim)，再将其扩为：(episode_num, n_master, obs_dim)
        # obs_mean = torch.mean(obs, dim=1, keepdim=True).expand(-1, self.n_master, -1)

        # slave原始维度(episode_num * n_slave, slave_hidden_dim), 变换后(episode_num, n_slave, slave_hidden_dim)
        slave_hidden_state = slave_hidden_state.view(episode_num, self.n_slave, -1)
        # 按第二个维度平均后：(episode_num, 1, slave_hidden_dim)，再将其扩为：(episode_num, n_master, slave_hidden_dim)
        hidden_mean = torch.mean(slave_hidden_state, dim=1, keepdim=True).expand(-1, self.n_master, -1)

        inputs = list()
        # 添加状态
        inputs.append(s)
        # 添加观测
        inputs.append(hidden_mean.cpu())

        # 将平均后的h和全局状态s拼接在一起（从list变为一个大Tensor）
        # 将维度从 (episode_num, n_master, inputs) 变为 (episode_num * n_master, inputs)
        inputs = torch.cat([x.reshape(episode_num * self.n_master, -1) for x in inputs], dim=1)

        if self.args.cuda:
            inputs = inputs.cuda()

        return inputs

    def _get_values(self, r, mask, terminated, max_episode_len):
        # r原始维度：(episode_num, max_episode_len, 1)
        r = r.squeeze(-1)  # 删除维数为1的维度
        mask = mask.squeeze(-1)  # 删除前：(episode_num, max_episode_len, 1)，删除后：(episode_num, max_episode_len)
        terminated = terminated.squeeze(-1)  # 只有episode结束的step才是1，其余全是0
        terminated = 1 - terminated  # 为了不影响价值的计算，将其改为1，最后一个为0
        n_return = torch.zeros_like(r)
        n_return[:, -1] = r[:, -1] * mask[:, -1]  # 将最后结束step的奖励直接取出，结束时刻的价值就是即时奖励
        for transition_idx in range(max_episode_len - 2, -1, -1):
            # 时间步从max_episode_len-2开始算是因为，最后一个step的价值已经得到不再需要计算，所以要从倒数第二个开始计算
            # v值的计算按照原有公式应为折扣奖励，但ms-marl论文中给出的公式并未做衰减，因此去掉了衰减因子gamma
            n_return[:, transition_idx] = (r[:, transition_idx] + n_return[:, transition_idx + 1] * terminated[:, transition_idx]) * mask[:, transition_idx]
        # 认为master与slave的v值一样，n_returns维度：(episode_num, max_episode_len, n_agents)
        return n_return.unsqueeze(-1).expand(-1, -1, self.n_agents)

    def _get_slave_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        # if self.args.master_index == 0:
        #     obs = batch["o"][:, transition_idx, 1:, :]
        # else:
        #     # master is the last one, to avoid the master obs
        #     obs = batch["o"][:, transition_idx, :self.args.master_index, :]
        # u_onehot = batch["u_onehot"][:]
        obs, u_onehot = batch["o"][:, transition_idx], batch["u_onehot"][:]

        episode_num = obs.shape[0]
        inputs = [obs]
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，则就让前一个动作为零向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
                # if self.args.master_index == 0:
                #     inputs.append(torch.zeros_like(u_onehot[:, transition_idx, 1:, :]))
                # else:
                #     inputs.append(torch.zeros_like(u_onehot[:, transition_idx, :self.args.master_index, :]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
                # if self.args.master_index == 0:
                #     inputs.append(u_onehot[:, transition_idx - 1, 1:, :])
                # else:
                #     inputs.append(u_onehot[:, transition_idx - 1, :self.args.master_index, :])
        if self.args.reuse_network:
            '''
            因为当前的inputs三维的数据，每一维分别代表(episode编号, agent编号, inputs维度)，直接在dim=1上添加对应的向量即可
            比如给agent_0后面加(1, 0, 0, 0, 0), 表示5个agent中的0号，而agent_0的数据正好在第0行，那么需要加的agent编号恰好
            就是一个单位阵，即对角线为1，其余为0
            '''
            inputs.append(torch.eye(self.n_slave).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条（40，96）的数据，
        # 因为这里所有智能体共享一个网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.n_slave, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch["o"].shape[0]  # batch中的episode数量
        avail_actions = batch["avail_u"]  # (episode_num, max_episode_len, n_agents, n_actions)
        action_prob = []  # 智能体采取的动作的概率
        for transition_idx in range(max_episode_len):
            inputs = self._get_slave_inputs(batch, transition_idx)  # 给obs加last_action、agent_id，获取slave的输入 (episode_num*n_slave, inputs_shape)
            if self.args.cuda:  # 是否使用gpu
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.h_master = self.h_master.cuda()
                self.c_master = self.c_master.cuda()
            # 由slave自身观测、slave隐藏状态的输入得到slave的输出和当前时刻的隐藏状态，outputs即为动作概率action_prob_ss此时还未进行归一化
            action_prob_ss, self.eval_hidden = self.slave(inputs, self.eval_hidden)
            # 将outputs维度重新变回(episode_num, n_slave, n_actions)，原始outputs维度为(episode_num * n_slave, n_actions)
            action_prob_ss = action_prob_ss.view(episode_num, self.n_slave, -1)

            # 得到master的输入数据
            master_inputs = self._get_master_inputs(batch, transition_idx, max_episode_len, self.eval_hidden)
            # 根据观测得到master的输出，m_outputs为master自身动作概率，action_prob_ms为master对slave指导的动作概率
            _, action_prob_ms, self.h_master, self.c_master = self.master(master_inputs, self.h_master, self.c_master, self.eval_hidden, self.n_slave)
            # transform the shape from (episode_num * n_slave, n_actions) to (episode_num, n_slave, n_actions)
            action_prob_ms = action_prob_ms.view(episode_num, self.n_slave, -1)

            slave_a_prob = action_prob_ss + action_prob_ms  # 相加得到slave的动作概率
            # # 将m_outputs维度变回(episode_num, n_master, n_action)，原始维度为(episode_num * n_master, n_actions)
            # m_outputs = m_outputs.view(episode_num, self.n_master, -1)

            # outputs = torch.cat((m_outputs, slave_a_prob), dim=1)  # 按第二个维度将master与slave的输出拼接起来，维度(episode_num, n_agents, n_actions)
            outputs = slave_a_prob  # slave agent action probability (episode_num, n_agents, n_actions)
            # if self.args.master_index == 0:  # master agent is the first one
            #     outputs = torch.cat((m_outputs, slave_a_prob), dim=1)  # 按第二个维度将master与slave的输出拼接起来，维度(episode_num, n_agents, n_actions)
            # else:  # master agent is the last one
            #     outputs = torch.cat((slave_a_prob, m_outputs), dim=1)  # 按第二个维度将master与slave的输出拼接起来，维度(episode_num, n_agents, n_actions)

            prob = torch.nn.functional.softmax(outputs, dim=-1)  # 输出经过softmax归一化
            action_prob.append(prob)  # all agent probability of actions at one time step
        # 得到的action_prob是一个列表，列表里装着max_episode_len个数组，数组每个元素的维度是(episode个数, n_agents, n_actions)
        # 把该列表转化成(episode个数, max_episode_len, n_agents, n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()

        # (episode_num, max_episode_len, n_agents, n_actions)
        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])  # 可选动作的个数
        # 选择动作时增加探索
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作置0

        # 上一步将不能执行动作置零，概率和不为1，需要重新正则化，执行过程中Categorical会自己正则化
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden, eval_hidden 隐藏状态
        self.eval_hidden = torch.zeros((episode_num, self.n_slave, self.args.slave_hidden_dim))
        # 为每个episode中的每个master都初始化一个hm和cm，即中间的隐藏状态
        self.h_master = torch.zeros((episode_num, self.n_master, self.args.master_hidden_dim))
        self.c_master = torch.zeros((episode_num, self.n_master, self.args.master_hidden_dim))

    # def init_master_hidden(self, episode_num):
    #     # 为每个episode中的每个master都初始化一个hm和cm，即中间的隐藏状态
    #     self.h_master = torch.zeros((episode_num, self.n_master, self.args.master_hidden_dim))
    #     self.c_master = torch.zeros((episode_num, self.n_master, self.args.master_hidden_dim))

    def save_model(self, run_num, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.slave.state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_slave_params.pkl")
        torch.save(self.master.state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_master_params.pkl")
