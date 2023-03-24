import numpy as np
import torch
from torch.distributions import one_hot_categorical
from common.utils import compute_gae, check, tensor2numpy
import random


class HAMSRolloutWorker:
    def __init__(self, agents, args):
        self.agents = agents
        self.eval_envs = args.eval_envs
        self.envs = args.envs
        self.episode_length = args.episode_limit
        # self.episode_length = args.episode_length
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.n_rollout_threads = args.n_rollout_threads
        self.n_eval_rollout_threads = args.n_eval_rollout_threads
        self.args = args

        # Noise linear decay
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        # Multi-scale
        self.top_k = args.top_k
        self.agent_scale = args.agent_scale
        self.candidate_actions = [[[None for i in range(j + 1)] for j in range(self.agent_scale)] for k in range(self.n_agents)]
        self.candidate_actions_log_prob = [[[None for i in range(j + 1)] for j in range(self.agent_scale)] for k in range(self.n_agents)]
        print('Init RolloutWorker')

    # Generate one episode for train
    @torch.no_grad()
    def generate_train_episode(self, noise_vector, episode_num=0, evaluate=False):
        o, u, u_prob, avail_u_prob, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], [], [], []
        values, log_probs, last_u_onehot = [], [], []
        self.envs.reset()
        step = 0  # time step
        episode_reward = 0  # cumulative reward
        last_action = np.zeros((self.n_rollout_threads, self.n_agents, self.n_actions))  # (n_threads, n_agents, n_actions)
        last_u_onehot.append(last_action.copy())
        self.agents.policy.model.init_hidden(1)

        # Epsilon decay
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while step < self.episode_length:
            obs = self.envs.get_obs()  # (n_threads, n_agents, obs_shape)
            state = self.envs.get_state()  # (n_threads, state_shape)
            actions, probs, avail_probs, avail_actions, actions_onehot = [], [], [], [], []
            action_log_probs = []
            value = None

            # Sample actions
            all_action, all_action_log_prob = None, None
            all_agents_avail_action = np.array([self.envs.get_avail_agent_actions(idx) for idx in range(self.n_agents)])  # (n_agents, n_threads, n_actions)
            all_agents_avail_action = np.array([all_agents_avail_action[:, thread_id, :] for thread_id in range(self.n_rollout_threads)])  # (n_threads, n_agents, n_actions)
            for agent_id in range(self.n_agents):
                agent_avail_action = self.envs.get_avail_agent_actions(agent_id)  # (n_threads, n_actions)
                if agent_id == 0:  # Calculate all agents strategy
                    all_action, all_action_log_prob, value = self.agents.hams_calculate_dist(state, obs, last_action,
                                                                                             all_agents_avail_action,
                                                                                             epsilon, noise_vector, evaluate)
                # Dim action: (n_threads,)  action_log_prob: (n_threads,)
                action, action_log_prob = self.agents.hams_choose_action(all_action, all_action_log_prob, agent_id, epsilon, evaluate)
                # if step == 0:
                #     action = all_action[:, agent_id].squeeze(-1)  # (n_threads,)
                #     action_log_prob = all_action_log_prob[:, agent_id].detach().squeeze(-1)  # (n_threads,)
                #     self.candidate_actions[agent_id] = [[action[j] for _ in range(j)] for j in range(self.agent_scale)]
                #     self.candidate_actions_log_prob[agent_id] = [[action_log_prob[j] for _ in range(j)] for j in range(self.agent_scale)]
                #     # print('aaaaaaaaaaaaa')
                # else:
                #     action, action_log_prob = [], []
                #     for threads_action_idx in range(len(self.candidate_actions[agent_id])):
                #         if not self.candidate_actions[agent_id][threads_action_idx]:  # Need to sample new action
                #             self.candidate_actions[agent_id][threads_action_idx] = [all_action[threads_action_idx, agent_id]
                #                                                                     for _ in range(threads_action_idx + 1)]
                #             self.candidate_actions_log_prob[agent_id][threads_action_idx] = [all_action_log_prob[threads_action_idx, agent_id]
                #                                                                              for _ in range(threads_action_idx + 1)]
                #         action_thread = self.candidate_actions[agent_id][threads_action_idx].pop()
                #         action_log_prob_thread = self.candidate_actions_log_prob[agent_id][threads_action_idx].pop()
                #         if agent_avail_action[threads_action_idx, action_thread] == 0:
                #             # The action is illegal
                #             action_thread = all_action[threads_action_idx, agent_id]
                #             action_log_prob_thread = all_action_log_prob[threads_action_idx, agent_id]
                #         action.append(action_thread.reshape(-1, 1))
                #         action_log_prob.append(action_log_prob_thread.reshape(-1, 1))
                #     action = torch.cat([x for x in action], dim=0).squeeze(-1)
                #     action_log_prob = torch.cat([x for x in action_log_prob], dim=0).squeeze(-1)

                # Generate onehot vector of the action
                action_onehot = np.zeros((self.n_rollout_threads, self.n_actions))
                actions.append(action.numpy().reshape(self.n_rollout_threads, 1))  # actions为各智能体在t时刻的动作
                action_log_probs.append(action_log_prob.numpy().reshape(self.n_rollout_threads, 1))  # 智能体t时刻动作概率对数
                actions_onehot.append(action_onehot)  # actions_onehot为各智能体在t时刻的动作onehot编码
                avail_actions.append(agent_avail_action)
                for thread in range(self.n_rollout_threads):
                    action_onehot[thread, action[thread]] = 1
                    last_action[thread, agent_id, :] = action_onehot[thread]

            # Transform data from (n_agents, n_threads, shape) to (n_threads, n_agents, shape)
            actions_t, action_log_probs_t, actions_onehot_t, avail_actions_t = [], [], [], []
            for thread in range(self.n_rollout_threads):
                actions_tmp = [x[thread, :] for x in actions]  # (n_agents, 1)
                action_log_probs_tmp = [x[thread, :] for x in action_log_probs]
                actions_onehot_tmp = [x[thread, :] for x in actions_onehot]
                avail_actions_tmp = [x[thread, :] for x in avail_actions]
                actions_t.append(actions_tmp)
                action_log_probs_t.append(action_log_probs_tmp)
                actions_onehot_t.append(actions_onehot_tmp)
                avail_actions_t.append(avail_actions_tmp)
            actions = torch.tensor(actions_t)  # (n_rollout_threads, n_agents, 1)
            action_log_probs = torch.tensor(action_log_probs_t)  # (n_rollout_threads, n_agents, 1)
            actions_onehot = torch.tensor(actions_onehot_t)  # (n_rollout_threads, n_agents, n_actions)
            avail_actions = torch.tensor(avail_actions_t)  # (n_rollout_threads, n_agents, n_actions)

            # Execute action
            # Dim: (n_threads,) (n_threads,)
            # info is tuple(5), each element is dict(3)
            reward, terminated, info = self.envs.step(actions)

            # Collect episode sample
            o.append(obs)
            s.append(state)
            u.append(tensor2numpy(actions))
            values.append(tensor2numpy(value))
            log_probs.append(tensor2numpy(action_log_probs))
            last_u_onehot.append(last_action.copy())
            u_onehot.append(tensor2numpy(actions_onehot))
            avail_u.append(tensor2numpy(avail_actions))
            r.append(reward.reshape(-1, 1))
            terminate.append(terminated.reshape(-1, 1))
            padded.append(np.zeros([self.n_rollout_threads, 1]))
            # padded.append([0.])
            episode_reward += reward

            step += 1
        # Final obs and state
        obs = self.envs.get_obs()
        state = self.envs.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # Get available action for final state, target_q needs avail_action in training
        final_avail_action = np.array([self.envs.get_avail_agent_actions(idx) for idx in range(self.n_agents)])  # (n_agents, n_threads, n_actions)
        final_avail_action = np.array([final_avail_action[:, thread_id, :] for thread_id in range(self.n_rollout_threads)])  # (n_threads, n_agents, n_actions)
        avail_u.append(final_avail_action)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        last_u_onehot = last_u_onehot[:-1]

        # Get next state value to use in GAE
        next_state = check(state).unsqueeze(1).repeat(1, self.n_agents, 1).reshape(-1, self.state_shape)
        if self.args.cuda:
            next_state = next_state.cuda()
        next_value = self.agents.policy.eval_critic(next_state, noise_vector).reshape(self.n_rollout_threads, self.n_agents, -1)  # (n_threads, n_agents, 1)
        next_value = tensor2numpy(next_value)

        # Compute returns and advantages by GAE
        returns = compute_gae(next_value, values, r, padded, self.args)
        advantages = np.stack(returns, axis=1) - np.stack(values, axis=1)

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       u_onehot=u_onehot.copy(),
                       avail_u=avail_u.copy(),
                       avail_u_next=avail_u_next.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy(),
                       values=values.copy(),
                       returns=returns.copy(),
                       advantages=advantages.copy(),
                       log_probs=log_probs.copy(),
                       last_u_onehot=last_u_onehot.copy()
                       )

        for key in episode.keys():
            if type(episode[key]) != np.ndarray:
                episode[key] = np.stack(episode[key], axis=1)
        if not evaluate:
            self.epsilon = epsilon

        # To exploration
        # if np.random.rand() <= self.epsilon:
        #     sample_idx = np.arange(self.agent_scale)
        #     top_k_reward_idx = np.random.choice(sample_idx, size=self.top_k, replace=False)
        # else:
        #     sort_reward_idx = np.argsort(episode_reward)
        #     top_k_reward_idx = sort_reward_idx[-self.top_k:]
        sort_reward_idx = np.argsort(episode_reward)
        top_k_reward_idx = sort_reward_idx[-self.top_k:]
        episode_sorted = dict()
        for key in episode.keys():
            episode_sorted[key] = []
            for choose_idx in top_k_reward_idx:
                episode_sorted[key].append(episode[key][choose_idx])
            episode_sorted[key] = np.stack(episode_sorted[key], axis=0)

        # return episode, episode_reward
        return episode_sorted, episode_reward

    # Generate one episode for evaluation
    @torch.no_grad()
    def generate_evaluate_episode(self, noise_vector, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and episode_num == 0 and evaluate:  # prepare for save replay of evaluation
            self.eval_envs.close()

        self.eval_envs.reset()
        step = 0
        terminated = [False]  # 终止标志
        eval_win_tag = False  # 获胜标志
        eval_episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.n_eval_rollout_threads, self.n_agents, self.n_actions))  # (n_threads, n_agents, n_actions)
        self.agents.policy.model.init_hidden(1)

        epsilon = 0
        while not terminated[0] and step < self.episode_length:
            obs = self.eval_envs.get_obs()  # (n_threads, n_agents, obs_shape)
            state = self.eval_envs.get_state()  # (n_threads, state_shape)
            actions = []

            # Sample actions
            all_action, all_action_log_prob = None, None
            all_agents_avail_action = np.array([self.eval_envs.get_avail_agent_actions(idx) for idx in range(self.n_agents)])  # (n_agents, n_threads, n_actions)
            all_agents_avail_action = np.array([all_agents_avail_action[:, thread_id, :] for thread_id in range(self.n_eval_rollout_threads)])
            for agent_id in range(self.n_agents):
                if agent_id == 0:
                    # Dim: (n_threads, n_agents, 1)
                    all_action, all_action_log_prob, _ = self.agents.hams_calculate_dist(state, obs, last_action,
                                                                                             all_agents_avail_action,
                                                                                             epsilon, noise_vector, evaluate)
                # Dim action: (n_threads,) action_log_prob: (n_threads,)
                action, action_log_prob = self.agents.hams_choose_action(all_action, all_action_log_prob, agent_id, epsilon, evaluate)

                # Generate onehot vector of actions
                action_onehot = np.zeros((self.n_eval_rollout_threads, self.n_actions))
                actions.append(action.numpy().reshape(self.n_eval_rollout_threads, 1))
                for thread_id in range(self.n_eval_rollout_threads):
                    action_onehot[thread_id, action[thread_id]] = 1
                    last_action[thread_id, agent_id, :] = action_onehot[thread_id]

            # Transform data from (n_agents, n_threads, shape) to (n_threads, n_agents, shape)
            actions_t = []
            for thread_id in range(self.n_eval_rollout_threads):
                actions_tmp = [x[thread_id, :] for x in actions]  # (n_agents, 1)
                actions_t.append(actions_tmp)
            actions = torch.tensor(actions_t)  # (n_threads, n_agents, 1)

            # Execute action
            # The type of all returns is np.array
            # Dim: (n_threads,) (n_threads,)
            # info is tuple (dim is n_threads), each element is dict (dim is n_agents). For single env, dim is (n_threads,)
            reward, terminated, info = self.eval_envs.step(actions)
            eval_win_tag = True if terminated[0] and 'battle_won' in info[0] and info[0]['battle_won'] else False

            eval_episode_reward += reward
            step += 1

        # End episode
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.eval_envs.save_replay()
            self.eval_envs.close()
        return eval_episode_reward, eval_win_tag
