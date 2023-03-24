import torch
import os
from network.hams_net import *
from network.hams_model import *
from common.utils import check


class HAMS:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        args.n_master, groups = self._get_master_number(args)  # 得到集群master的数量和集群总数
        args.n_slaves = [n for n in groups.values()]  # 每个集群内智能体数量就是slave的数量
        self.n_master = args.n_master  # the master is virtual, each group has a master
        self.n_slaves = args.n_slaves  # 一个列表，取值时要注意使用索引
        self.groups = groups
        self.args = args
        master_input_shape = self._get_master_input_shape()  # master网络的输入维度
        slaves_input_shape = self._get_slaves_input_shape()  # 列表，各集群内slave网络的输入维度
        gcms_input_shape = self._get_gcms_input_shape()  # GCM网络的输入维度
        commgroup_input_shape = self._get_commgroup_input_shape()  # 集群间通信网络的输入维度
        critic_input_shape = self._get_critic_input_shape()  # critic网络的输入维度

        self.slaves = []  # 各集群的slave
        self.masters = []  # 各集群的master
        self.gcms = []  # 各集群的GCM
        self.commGroup = None  # 集群间通信网络
        if self.args.alg == "hams":
            print("Init alg hams")
            for i in range(self.n_master):  # 构建各集群的master和slave网络
                self.slaves.append(CommSlave(slaves_input_shape[i], args))  # 集群内slave共用一个网络，集群间slave网络结构相同但参数不一样
                # self.masters.append(CommMaster(master_input_shape, args))  # 集群间master网络结构相同，但参数不同
                # self.gcms.append(GCM(gcms_input_shape, args))  # GCM主要配合master对slave的动作进行指导
            # self.commGroup = CommGroup(commgroup_input_shape, args)  # 集群间使用一个网络进行通信
        else:
            raise Exception("No such algorithm")

        # 得到当前状态下的状态价值估计作为baseline
        self.eval_critic = Critic(critic_input_shape, args)  # 价值网络

        # 是否使用GPU，服务器上可以使用GPU，window下建议CPU
        if self.args.cuda:
            for i in range(self.n_master):
                self.slaves[i].cuda()
                # self.masters[i].cuda()
                # self.gcms[i].cuda()
            # self.commGroup.cuda()
            self.eval_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            file_names = os.listdir(self.model_dir)  # List of all filename in current path
            if file_names:
                # Load the last model weights saved in the final round (maybe not optimal) for HAMS
                model_weights_paths = file_names[-8:]  # Read the last eight datas as weights
                path_commgroup = self.model_dir + "/" + model_weights_paths[0]
                path_critic = self.model_dir + "/" + model_weights_paths[1]
                map_location = "cuda:0" if self.args.cuda else "cpu"
                file_idx = 2
                for i in range(self.n_master):
                    path_gcm = self.model_dir + "/" + model_weights_paths[file_idx]
                    file_idx = file_idx + 1
                    path_master = self.model_dir + "/" + model_weights_paths[file_idx]
                    file_idx = file_idx + 1
                    path_slave = self.model_dir + "/" + model_weights_paths[file_idx]
                    file_idx = file_idx + 1
                    # self.gcms[i].load_state_dict(torch.load(path_gcm, map_location=map_location))
                    # self.masters[i].load_state_dict(torch.load(path_master, map_location=map_location))
                    self.slaves[i].load_state_dict(torch.load(path_slave, map_location=map_location))
                # self.commGroup.load_state_dict(torch.load(path_commgroup, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_critic, map_location=map_location))
                print("Successfully load the model!")
            else:
                raise Exception("No model!")

        # 算法的各网络参数
        self.slaves_parameters = []
        # self.masters_parameters = []
        # self.gcms_parameters = []
        for i in range(self.n_master):
            self.slaves_parameters.append(list(self.slaves[i].parameters()))
            # self.masters_parameters.append(list(self.masters[i].parameters()))
            # self.gcms_parameters.append(list(self.gcms[i].parameters()))
        # self.commGroup_parameters = list(self.commGroup.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        # 算法各网络对应的优化器
        self.slaves_optimizer = []
        # self.masters_optimizer = []
        # self.gcms_optimizer = []
        if args.optimizer == "RMS":
            for i in range(self.n_master):
                self.slaves_optimizer.append(torch.optim.RMSprop(self.slaves_parameters[i], lr=args.lr_slave))
                # self.masters_optimizer.append(torch.optim.RMSprop(self.masters_parameters[i], lr=args.lr_master))
                # self.gcms_optimizer.append(torch.optim.RMSprop(self.gcms_parameters[i], lr=args.lr_gcm))
            # self.commGroup_optimizer = torch.optim.RMSprop(self.commGroup_parameters, lr=args.lr_comm)
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
        if args.optimizer == "ADAM":
            for i in range(self.n_master):
                self.slaves_optimizer.append(torch.optim.Adam(self.slaves_parameters[i], lr=args.lr_slave))
                # self.masters_optimizer.append(torch.optim.Adam(self.masters_parameters[i], lr=args.lr_master))
                # self.gcms_optimizer.append(torch.optim.Adam(self.gcms_parameters[i], lr=args.lr_gcm))
            # self.commGroup_optimizer = torch.optim.Adam(self.commGroup_parameters, lr=args.lr_comm)
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=args.lr_critic)

        # learning rate schedulers for correlated networks except the critic network
        self.slaves_lr_scheduler = []
        # self.masters_lr_scheduler = []
        # self.gcms_lr_scheduler = []
        # self.commGroup_lr_scheduler = None
        for i in range(self.n_master):
            self.slaves_lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.slaves_optimizer[i],
                                                                            step_size=self.args.lr_step_size,
                                                                            gamma=self.args.lr_gamma))
            # self.masters_lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.masters_optimizer[i],
            #                                                                  step_size=self.args.lr_step_size,
            #                                                                  gamma=self.args.lr_gamma))
            # self.gcms_lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.gcms_optimizer[i],
            #                                                               step_size=self.args.lr_step_size,
            #                                                               gamma=self.args.lr_gamma))
        # self.commGroup_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.commGroup_optimizer,
        #                                                               step_size=self.args.lr_step_size,
        #                                                               gamma=self.args.lr_gamma)

        # # 为每个集群的slave维护一个eval_hidden，故是一个列表
        # self.eval_hidden = []
        # # 同self.eval_hidden，master为LSTM故需要h和c
        # self.h_master = []
        # self.c_master = []
        # # GCM的参数由master和slave输入不需要维护，commGroup只有一个，也需要维护h和c
        # self.h_commGroup = None
        # self.c_commGroup = None

        # 构建算法的模型
        self.model = HamsModel(args, self.slaves, self.masters, self.gcms, self.commGroup, self.eval_critic)

    @staticmethod
    def _get_master_number(args):
        """
        获取地图中master的数量，由于一个集群只有一个master，故master数量与集群数量一致
        输出为master的数量，以及所有集群智能体数量构成的字典group，group内为各类智能体的数量
        """
        last_unit_type = 0  # 上一个的单位类型
        n_master = 0  # master的数量
        n_group = 0  # 集群内智能体的数量（包括master和slave）
        group = dict()  # 记录各集群的智能体数量
        for key, value in args.env.agents.items():
            if value.unit_type != last_unit_type:  # 当前的unit_type与上一次的不一致时已经是另一个集群
                last_unit_type = value.unit_type
                n_master += 1
                n_group = 0  # 当开始计算下一个集群时，将智能体数量清空
            n_group += 1  # 集群内智能体数量+1
            group[last_unit_type] = n_group

        return n_master, group

    def _get_master_input_shape(self):
        # slave的隐藏状态h_s
        input_shape = self.args.slave_hidden_dim
        # Attention网络中的隐层维度，代码中所有attention使用的网络维度一致都是attention_dim
        input_shape += self.args.attention_dim

        return input_shape

    def _get_slaves_input_shape(self):
        slave_input_shape = []
        for i in range(len(self.args.n_slaves)):
            s_input_shape = self.obs_shape  # slave网络的输入维度和coma、vdn、qmix的rnn一致
            # 根据参数决定slave的输入维度
            if self.args.last_action:
                s_input_shape += self.args.n_actions
            if self.args.reuse_network:
                s_input_shape += self.args.n_slaves[i]
            slave_input_shape.append(s_input_shape)  # 得到各集群的slave输入维度

        return slave_input_shape

    def _get_gcms_input_shape(self):
        # slave的隐藏状态h_s
        input_shape = self.args.slave_hidden_dim
        # Attention网络中的隐层维度，代码中所有attention使用的网络维度一致都是attention_dim
        input_shape += self.args.attention_dim

        return input_shape

    def _get_commgroup_input_shape(self):
        input_shape = self.args.master_hidden_dim

        return input_shape

    def _get_critic_input_shape(self):
        # 输入全局状态
        if self.args.use_value_noise:
            input_shape = self.state_shape + self.args.noise_dim
        else:
            input_shape = self.state_shape

        return input_shape

    def learn(self, batch, max_episode_len, train_step, noise_vector, epsilon):
        """
        batch中每个数据都是四维，第一个维度为batch_size表示batch中有几个episode，第二个维度表示batch中的episode长度，
        第三个维度表示智能体数量，第四个维度表示具体的数据的维度(obs,state,actions等)
        """
        # batch_size = batch['o'].shape[0]  # batch_size
        # self.model.init_hidden(batch_size)  # 为master、slave、gcm以及commGroup初始化隐藏状态
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        # u, r, avail_u, terminated = batch['u'], batch['r'], batch["avail_u"], batch["terminated"]
        # u_prob = batch['u_prob']  # 动作对应的概率 (batch_size, max_episode_len, n_agents, 1)
        # avail_u_prob = batch['avail_u_prob']  # 可用动作的概率 (batch_size, max_episode_len, n_agents, n_actions)
        # mask维度: (batch_size, max_episode_len, 1)与batch["padded"]维度一致，mask为1，padded为0
        # mask = (1 - batch["padded"].float())  # 用来把那些填充的经验置0，从而不让他们影响到学习
        # if self.args.cuda:
            # r = r.cuda()
            # u = u.cuda()
            # mask = mask.cuda()
            # terminated = terminated.cuda()  # (batch_size, max_episode_len, 1)，终止step及之后为1，其余为0
            # u_prob = u_prob.cuda()
            # avail_u_prob = avail_u_prob.cuda()

        batch_adv = batch["advantages"]
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-5)
        batch["advantages"] = batch_adv

        for _ in range(self.args.ppo_epochs):
            for sample_data in self.model.ppo_generator(batch, self.args.num_mini_batch):
                # with torch.autograd.set_detect_anomaly(True):
                # execute a single forward pass for mini_batch steps
                distributions, values = self.model(sample_data, epsilon, noise_vector)  # ---add
                values = values.reshape(-1, self.n_agents, 1)

                state_batch, obs_batch, available_actions_batch, last_actions_batch, actions_batch, \
                value_preds_batch, returns_batch, advantages_batch, old_action_log_probs_batch, padded_batch = \
                    sample_data["s"], sample_data["o"], sample_data["avail_u"], sample_data["last_u_onehot"], sample_data["u"], \
                    sample_data["values"], sample_data["returns"], sample_data["advantages"], sample_data["log_probs"], sample_data["padded"]

                mask_batch = (1 - padded_batch.float())  # (mini_batch_size, n_agents, 1)
                dist_entropy = distributions.entropy().mean()  # a number
                new_action_log_probs_batch = distributions.log_prob(actions_batch.reshape(-1, 1).squeeze(-1))
                new_action_log_probs_batch = new_action_log_probs_batch.unsqueeze(-1).reshape(-1, self.n_agents, 1)  # (mini_batch_size, n_agents, 1)

                if self.args.use_adv_noise:  # ---add
                    adv_noise = check(noise_vector).unsqueeze(0)
                    adv_noise = adv_noise[:, :, 0:1].repeat(advantages_batch.shape[0], 1, 1)
                    advantages_batch = (1 - self.args.alpha) * advantages_batch + self.args.alpha * adv_noise

                if self.args.cuda:
                    mask_batch = mask_batch.cuda()
                    dist_entropy = dist_entropy.cuda()
                    returns_batch = returns_batch.cuda()
                    advantages_batch = advantages_batch.cuda()
                    value_preds_batch = value_preds_batch.cuda()
                    new_action_log_probs_batch = new_action_log_probs_batch.cuda()
                    old_action_log_probs_batch = old_action_log_probs_batch.cuda()

                ratio = torch.exp(new_action_log_probs_batch - old_action_log_probs_batch)
                surr1 = ratio * advantages_batch  # (mini_batch_size, n_agents, 1)
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages_batch

                # Actor update
                policy_loss = -(torch.min(surr1, surr2) * mask_batch).sum() / mask_batch.sum()
                # 梯度清零
                for i in range(self.n_master):
                    self.slaves_optimizer[i].zero_grad()
                    # self.masters_optimizer[i].zero_grad()
                    # self.gcms_optimizer[i].zero_grad()
                # self.commGroup_optimizer.zero_grad()
                (policy_loss - dist_entropy * self.args.entropy_coef).backward()  # loss反向传播
                # normalize the parameters gradients
                for i in range(self.n_master):
                    torch.nn.utils.clip_grad_norm_(self.slaves_parameters[i], self.args.grad_norm_clip)
                    # torch.nn.utils.clip_grad_norm_(self.masters_parameters[i], self.args.grad_norm_clip)
                    # torch.nn.utils.clip_grad_norm_(self.gcms_parameters[i], self.args.grad_norm_clip)
                # torch.nn.utils.clip_grad_norm_(self.commGroup_parameters, self.args.grad_norm_clip)
                # 将对应梯度更新到网络参数上
                for i in range(self.n_master):
                    self.slaves_optimizer[i].step()
                    # self.masters_optimizer[i].step()
                    # self.gcms_optimizer[i].step()
                # self.commGroup_optimizer.step()
                # learning rate decay
                for i in range(self.n_master):
                    self.slaves_lr_scheduler[i].step()
                    # self.masters_lr_scheduler[i].step()
                    # self.gcms_lr_scheduler[i].step()
                # self.commGroup_lr_scheduler.step()

                # Critic update
                error = ((returns_batch - values) ** 2) / 2
                # error_original = ((returns_batch - values) ** 2) / 2
                # value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.args.clip_param, self.args.clip_param)
                # error_clipped = ((returns_batch - value_pred_clipped) ** 2) / 2
                # error = torch.max(error_original, error_clipped)
                value_loss = (error * mask_batch).sum() / mask_batch.sum()
                self.critic_optimizer.zero_grad()
                (value_loss * self.args.value_loss_coef).backward()
                torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
                self.critic_optimizer.step()

        # 训练critic网络，并且得到每条经验的td_error (batch_size, max_episode_len, 1)
        # td_error = self._train_critic(batch, max_episode_len, train_step)
        # td_error = td_error.repeat(1, 1, self.n_agents).detach()

        # # 得到每条经验的return，(batch_size, max_episode_len, n_agents)
        # values = self._get_values(r, mask, terminated, max_episode_len)

        # # 每个智能体的所有动作的概率，(batch_size, max_episode_len, n_agents, n_actions)
        # action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        # # mask维度三维，repeat函数在第一个维度上复制为1个，在第二个维度上复制为1个，在第三个维度上复制为智能体数量
        # # 给mask转换出n_agents维度，用于每个agent的训练
        # mask = mask.repeat(1, 1, self.n_agents)  # (batch_size, max_episode_len, n_agents)

        # pi_taken = u_prob.squeeze(-1)  # after squeeze pi_taken: (batch_size, max_episode_len, n_agents)
        # # pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # (batch_size, max_episode_len, n_agents)
        # pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验概率为0，取log则为负无穷，故赋1取对数后为0
        # log_pi_taken = torch.log(pi_taken)

        # 策略梯度的方法更新智能体参数
        # 将总loss均分到每个智能体上
        # total_loss = -((td_error * log_pi_taken) * mask).sum() / mask.sum()
        # # 梯度清零
        # for i in range(self.n_master):
        #     self.slaves_optimizer[i].zero_grad()
        #     self.masters_optimizer[i].zero_grad()
        #     self.gcms_optimizer[i].zero_grad()
        # self.commGroup_optimizer.zero_grad()
        # total_loss.backward()  # loss反向传播
        # # normalize the parameters gradients
        # for i in range(self.n_master):
        #     torch.nn.utils.clip_grad_norm_(self.slaves_parameters[i], self.args.grad_norm_clip)
        #     torch.nn.utils.clip_grad_norm_(self.masters_parameters[i], self.args.grad_norm_clip)
        #     torch.nn.utils.clip_grad_norm_(self.gcms_parameters[i], self.args.grad_norm_clip)
        # torch.nn.utils.clip_grad_norm_(self.commGroup_parameters, self.args.grad_norm_clip)
        # # 将对应梯度更新到网络参数上
        # for i in range(self.n_master):
        #     self.slaves_optimizer[i].step()
        #     self.masters_optimizer[i].step()
        #     self.gcms_optimizer[i].step()
        # self.commGroup_optimizer.step()
        # # learning rate decay
        # for i in range(self.n_master):
        #     self.slaves_lr_scheduler[i].step()
        #     self.masters_lr_scheduler[i].step()
        #     self.gcms_lr_scheduler[i].step()
        # self.commGroup_lr_scheduler.step()

    # def _get_master_inputs(self, batch, transition_idx, slave_hidden_state):
    #     """
    #     此函数为计算所有集群master的输入。batch为所有集群的输入，计算时需要对集群进行区分，取出不同集群的样本信息进行计算。
    #     batch维度：(batch_size, max_episode_len, n_agents, shape)
    #     slave_hidden_state为一个列表，每个元素维度：(batch_size * n_slave, slave_hidden_dim + attention_dim)，n_slave为组内slave智能体个数
    #     """
    #     # # transition_idx为一个episode中的第几个step, max_episode_len为batch中episode长度最大值
    #     # # 对于每个集群来说，认为集群内第一个智能体为master，master有全局状态不需要自身观测，要取其他slave的隐藏状态要从1开始
    #     # obs = []
    #     # obs_next = []
    #     # index = 0
    #     # for i in range(len(self.n_slaves)):
    #     #     # 列表内元素为此集群内slave的数量，加1便是该集群的智能体个数（master为1个）
    #     #     group_agents = self.n_slaves[i] + 1  # 得到该集群内智能体数量
    #     #     index += group_agents  # 累计已经走过的集群智能体数量
    #     #     if i == 0:  # 第一个集群的第一个为master，故从1开始取
    #     #         obs.append(batch["o"][:, transition_idx, 1:index])
    #     #         obs_next.append(batch["o_next"][:, transition_idx, 1:index])
    #     #     else:  # 此后的集群第一个都是master，所以要在1上加上已经走过的智能体数量作为起始索引
    #     #         obs.append(batch["o"][:, transition_idx, 1 + index - group_agents:index])
    #     #         obs_next.append(batch["o_next"][:, transition_idx, 1 + index - group_agents:index])
    #     s, s_next = batch["s"][:, transition_idx], batch["s_next"][:, transition_idx]
    #
    #     # batch中s为(batch_size, max_episode_len, state_shape)，此函数中的s为从第二维度取出某一个时间步后，故不是三维
    #     # s是二维没有n_agents维度，因为所有agent的s一样，其他数据都是三维不能拼接，所以要把s转化为三维: (batch_size, n_master, state_shape)
    #     s = s.unsqueeze(1).expand(-1, self.n_master, -1)
    #
    #     # for i in range(len(slave_hidden_state)):
    #     #     # 变换维度从(batch_size * n_slave, slave_hidden_dim + attention_dim)到(batch_size, n_slave, slave_hidden_dim + attention_dim)
    #     #     slave_hidden_state[i] = slave_hidden_state[i].view(episode_num, self.n_slaves[i], -1)
    #
    #     if self.args.cuda:
    #         s = s.cuda()
    #         for i in range(len(slave_hidden_state)):
    #             slave_hidden_state[i] = slave_hidden_state[i].cuda()
    #
    #     # master的输入除了全局观测外，还有自己组内slave的隐藏状态
    #     # s的维度是(batch_size, n_master, state_shape)
    #     # slave_hidden_state为各集群内状态，是一个列表，每个元素维度是(batch_size, n_slave, slave_hidden_dim + attention_dim)
    #     return s, slave_hidden_state
    #
    # def _get_values(self, r, mask, terminated, max_episode_len):
    #     # r原始维度：(batch_size, max_episode_len, 1)
    #     r = r.squeeze(-1)  # 删除维度为1的维度 (batch_size, max_episode_len)
    #     mask = mask.squeeze(-1)  # 删除前：(batch_size, max_episode_len, 1)，删除后：(batch_size, max_episode_len)
    #     terminated = terminated.squeeze(-1)  # 只有episode结束的step及往后才是1，其余全是0
    #     terminated = 1 - terminated  # 为了不影响价值的计算，将其改为1，最后一个为0
    #     n_return = torch.zeros_like(r)
    #     n_return[:, -1] = r[:, -1] * mask[:, -1]  # 将最后结束step的奖励直接取出，结束时刻的价值就是即时奖励
    #     for transition_idx in range(max_episode_len - 2, -1, -1):
    #         # 时间步从max_episode_len-2开始算是因为，最后一个step的价值已经得到不再需要计算，所以要从倒数第二个开始计算
    #         # v值的计算按照原有公式应为折扣奖励
    #         n_return[:, transition_idx] = (r[:, transition_idx] + self.args.gamma * n_return[:, transition_idx + 1] * terminated[:, transition_idx]) * mask[:,
    #                                                                                                                                                    transition_idx]
    #
    #     # 认为所有智能体的v值一样，n_return维度：(batch_size, max_episode_len, n_agents)
    #     return n_return.unsqueeze(-1).expand(-1, -1, self.n_agents)
    #
    # def _get_slave_inputs(self, batch, transition_idx):
    #     """
    #     此函数为计算所有集群的slave输入。batch为所有集群的输入，计算时需要对集群进行区分，取出不同集群的样本信息进行计算
    #     """
    #     # 取出所有episode上该transition_idx的经验
    #     # obs与u_onehot均为列表，obs内元素维度(batch_size, n_slave, state_shape)，u_onehot内元素维度(batch_size, max_episode_len, n_slave, n_actions)
    #     obs, u_onehot = [], []
    #     index = 0
    #     # 取出对应的样本
    #     for i in range(len(self.n_slaves)):
    #         group_agents = self.n_slaves[i]
    #         index += group_agents
    #         obs.append(batch["o"][:, transition_idx, index - group_agents:index])
    #         u_onehot.append(batch["u_onehot"][:, :, index - group_agents:index])
    #
    #     episode_num = batch["o"].shape[0]
    #     inputs = []  # 所有集群slave的输入，是一个列表，每个元素维度(batch_size * n_slave, obs_shape + n_action + n_slave)
    #     for i in range(len(self.n_slaves)):
    #         input_i = list()
    #         input_i.append(obs[i])  # 集群i内slave的输入，给input_i加上一个动作、agent编号
    #         if self.args.last_action:
    #             if transition_idx == 0:  # 如果是第一条经验，则就让前一个动作为零向量
    #                 input_i.append(torch.zeros_like(u_onehot[i][:, transition_idx]))
    #             else:
    #                 input_i.append(u_onehot[i][:, transition_idx - 1])
    #         if self.args.reuse_network:
    #             '''
    #             因为当前的inputs三维的数据，每一维分别代表(episode编号, agent编号, inputs维度)，直接在dim=1上添加对应的向量即可
    #             比如给agent_0后面加(1, 0, 0, 0, 0), 表示5个agent中的0号，而agent_0的数据正好在第0行，那么需要加的agent编号恰好
    #             就是一个单位阵，即对角线为1，其余为0
    #             '''
    #             input_i.append(torch.eye(self.n_slaves[i]).unsqueeze(0).expand(episode_num, -1, -1))
    #         # 要把inputs中的三个拼起来，因为群内slave共享一个网络，每条数据中带上了自己的编号，所以还是自己的数据
    #         input_i = torch.cat([x.reshape(episode_num * self.n_slaves[i], -1) for x in input_i], dim=1)  # (batch_size*n_slave, obs_shape+n_action+n_slave)
    #         inputs.append(input_i)
    #
    #     return inputs
    #
    # def _get_action_prob(self, batch, max_episode_len, epsilon):
    #     """
    #     此函数为计算所有智能体的动作概率。batch为所有集群的输入样本，计算时先需要对集群进行区分，取出不同集群的样本信息进行计算
    #     """
    #     episode_num = batch["o"].shape[0]  # batch_size，即batch中episode数量
    #     avail_actions = batch["avail_u"]  # (batch_size, max_episode_len, n_agent, n_action)
    #     action_prob = []  # 智能体采取的动作的概率
    #     for transition_idx in range(max_episode_len):
    #         # 给obs加last_action、agent_id，获取各集群slave的输入，列表元素维度(batch_size * n_slave, input_shape)
    #         slave_inputs = self._get_slave_inputs(batch, transition_idx)
    #         if self.args.cuda:
    #             for i in range(self.n_master):
    #                 slave_inputs[i] = slave_inputs[i].cuda()
    #                 self.eval_hidden[i] = self.eval_hidden[i].cuda()
    #                 self.h_master[i] = self.h_master[i].cuda()
    #                 self.c_master[i] = self.c_master[i].cuda()
    #             self.h_commGroup = self.h_commGroup.cuda()
    #             self.c_commGroup = self.c_commGroup.cuda()
    #         # 将slave自身观测、隐藏状态输入得到slave输出和当前时刻的隐藏状态，输出第一个为智能体自身策略，第二个为h_slave，第三个为加入其他信息的h_slave'
    #         action_prob_ss, new_h_slave = [], []
    #         for i in range(len(self.n_slaves)):
    #             # action_prob_i: (batch_size * n_slave, n_actions)
    #             # self.eval_hidden[i]: (batch_size * n_slave, slave_hidden_dim)
    #             # new_h_slave_i: (batch_size * n_slave, slave_hidden_dim + attention_dim)
    #             action_prob_i, self.eval_hidden[i], new_h_slave_i = self.slaves[i](slave_inputs[i], self.eval_hidden[i], self.n_slaves[i])
    #             action_prob_ss.append(action_prob_i.view(episode_num, self.n_slaves[i], -1))  # 变换维度并保存 (batch_size, n_slave, n_actions)
    #             new_h_slave.append(new_h_slave_i)
    #
    #         # 得到master的输入数据，m_s维度(batch_size, n_master, state_shape)，第二个为列表每个元素维度(batch_size, n_slave, slave_hidden_dim + attention_dim)
    #         master_states, h_s = self._get_master_inputs(batch, transition_idx, new_h_slave)
    #         for i in range(self.n_master):
    #             # self.h_master[i]: (batch_size, 1, master_hidden_dim), self.c_master[i]: (batch_size, 1, master_hidden_dim)
    #             # 由全局状态和h_s得到master的h_m和c_m，后续还要经过commGroup、GCM两部分才能得到master的动作策略和其对slave的指导策略
    #             self.h_master[i], self.c_master[i] = self.masters[i](master_states[:, i], self.h_master[i], self.c_master[i], h_s[i])
    #
    #         # CommGroup 将h_m进行拼接(batch_size, n_master, master_hidden_dim)
    #         comm_inputs = torch.stack(self.h_master, dim=1)
    #         comm_inputs = comm_inputs.reshape(episode_num * self.n_master, -1)  # (batch_size * n_master, master_hidden_dim)
    #         # m_actions: (batch_size * n_master, n_actions)，h_m'和c_m'维度(batch_size * n_master, gcm_hidden_dim)送入GCM
    #         _, self.h_commGroup, self.c_commGroup = self.commGroup(comm_inputs, self.h_commGroup, self.c_commGroup)
    #         self.h_commGroup = self.h_commGroup.view(episode_num, self.n_master, -1)  # (batch_size, n_master, gcm_hidden_dim)
    #         self.c_commGroup = self.c_commGroup.view(episode_num, self.n_master, -1)  # (batch_size, n_master, gcm_hidden_dim)
    #
    #         # GCM 每个master的h_m'和c_m'送入GCM计算对组内slave的指导动作策略
    #         action_prob_ms = []
    #         for i in range(self.n_master):
    #             # (batch_size * n_slave, n_actions)
    #             action_prob_j = self.gcms[i](self.h_commGroup[:, i], self.c_commGroup[:, i], new_h_slave[i], self.n_slaves[i])
    #             action_prob_ms.append(action_prob_j.view(episode_num, self.n_slaves[i], -1))
    #
    #         # m_actions = m_actions.view(episode_num, self.n_master, -1)  # (batch_size, n_master, n_actions)
    #         s_actions = []
    #         for i in range(len(self.n_slaves)):
    #             s_actions.append(action_prob_ss[i] + action_prob_ms[i])  # 相加得到slave的动作概率，元素维度(batch_size, n_slave, n_actions)
    #
    #         # output = []  # 所有智能体动作策略，列表元素维度(batch_size, n_slave, n_actions)
    #         # for i in range(len(self.n_slaves)):
    #         #     m_output = m_actions[:, i].unsqueeze(1)  # (batch_size, 1, n_actions)
    #         #     output_i = torch.cat([m_output, s_actions[i]], dim=1)  # 将集群i内master和slave的动作策略拼接起来 (batch_size, n_slave, n_actions)
    #         #     output.append(output_i)  # 保存集群i所有智能体的动作策略
    #         output = torch.cat(s_actions, dim=1)  # 将各集群智能体动作策略拼接起来(batch_size, n_agents, n_actions)
    #         prob = torch.nn.functional.softmax(output, dim=-1)  # 输出经过softmax归一化
    #         action_prob.append(prob)  # 在当前时间步所有智能体的动作概率
    #     # 得到的action_prob是一个列表，列表里装着max_episode_len个数组，数组每个元素的维度(batch_size, n_agents, n_actions)
    #     # 把该列表转化成(batch_size, max_episode_len, n_agents, n_actions)的数组
    #     action_prob = torch.stack(action_prob, dim=1).cpu()
    #
    #     # 可选动作的个数 (batch_size, max_episode_len, n_agents, n_actions)
    #     action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])
    #     # 选动作时增加探索
    #     action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
    #     action_prob[avail_actions == 0] = 0.0  # 不能执行的动作置0
    #
    #     # 上一步将不能执行动作置零，概率和不为1，需要重新正则化，执行过程中Categorical会自己正则化
    #     action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
    #     # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan
    #     # 因此需要再一次将该经验对应的概率置为0
    #     action_prob[avail_actions == 0] = 0.0
    #     if self.args.cuda:
    #         action_prob = action_prob.cuda()
    #     return action_prob

    # def _get_v_values(self, batch, max_episode_len):
    #     v_evals, v_targets = [], []
    #     for transition_idx in range(max_episode_len):
    #         inputs, inputs_next = batch["s"][:, transition_idx], batch["s_next"][:, transition_idx]
    #         if self.args.cuda:
    #             inputs = inputs.cuda()
    #             inputs_next = inputs_next.cuda()
    #         # 价值网络和目标网络输入均为(batch_size, state_shape)，输出(batch_size, 1)
    #         v_eval = self.eval_critic(inputs)
    #         v_target = self.target_critic(inputs_next)
    #         v_evals.append(v_eval)
    #         v_targets.append(v_target)
    #     v_evals = torch.stack(v_evals, dim=1)  # 按时间维度叠加(batch_size, max_episode_len, 1)
    #     v_targets = torch.stack(v_targets, dim=1)  # 按时间维度叠加(batch_size, max_episode_len, 1)
    #     return v_evals, v_targets
    #
    # def _train_critic(self, batch, max_episode_len, train_step):
    #     r, terminated = batch["r"], batch["terminated"]
    #     mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
    #     if self.args.cuda:
    #         mask = mask.cuda()
    #         r = r.cuda()
    #         terminated = terminated.cuda()
    #     # 从全局状态分别得到价值网络、和目标网络的v值
    #     v_evals, v_next_target = self._get_v_values(batch, max_episode_len)
    #
    #     # Calculate the TD targets and TD error
    #     targets = r + self.args.gamma * v_next_target * (1 - terminated)
    #     td_error = targets.detach() - v_evals
    #     mask_td_error = mask * td_error  # 对td_error抹掉填充的经验
    #
    #     # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
    #     loss = (mask_td_error ** 2).sum() / mask.sum()
    #     # print("Critic Loss is ", loss)
    #     self.critic_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
    #     self.critic_optimizer.step()
    #     if train_step > 0 and train_step % self.args.target_update_cycle == 0:
    #         self.target_critic.load_state_dict(self.eval_critic.state_dict())  # hard update
    #     return td_error

    # def init_hidden(self, episode_num, episode_len):
    #     """
    #     为所有网络初始化隐藏状态变量（使用的RNN结构）
    #     """
    #     # 初始化变量前先清空
    #     self.eval_hidden.clear()
    #     self.h_master.clear()
    #     self.c_master.clear()
    #
    #     for i in range(self.n_master):
    #         self.eval_hidden.append(torch.zeros((episode_num, episode_len, self.n_slaves[i], self.args.slave_hidden_dim)))
    #         self.h_master.append(torch.zeros((episode_num, episode_len, 1, self.args.master_hidden_dim)))
    #         self.c_master.append(torch.zeros((episode_num, episode_len, 1, self.args.master_hidden_dim)))
    #     self.h_commGroup = torch.zeros((episode_num, episode_len, self.n_master, self.args.gcm_hidden_dim))
    #     self.c_commGroup = torch.zeros((episode_num, episode_len, self.n_master, self.args.gcm_hidden_dim))

    def save_model(self, run_num, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        for i in range(self.n_master):
            # 文件名保存格式，eg:r_n_g1_slave_params.pkl，r为第几次run，n为一次run中第几次保存，g1表示集群1，slave_params为slave的网络参数
            torch.save(self.slaves[i].state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_g" + str(i) + "_slave_params.pkl")
            # torch.save(self.masters[i].state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_g" + str(i) + "_master_params.pkl")
            # torch.save(self.gcms[i].state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_g" + str(i) + "_gcm_params.pkl")
        # commGroup只有一个，故不需要加集群编号:r_n_commGroup_params.pkl
        # torch.save(self.commGroup.state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_commGroup_params.pkl")
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + str(run_num) + "_" + num + "_critic_params.pkl")
