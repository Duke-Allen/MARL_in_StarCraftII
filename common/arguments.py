import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # parser.add_argument('--replay_dir', type=str, default='F:/ProjectCode/PycharmProjects/StarCraft2_v1.6-half-test/replay', help='absolute path to save the replay')
    # parser.add_argument('--replay_dir', type=str, default='/home/ymr/PycharmProjects/StarCraft2_v1.6-half-test/replay', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven, msmarl, msmarl+communication, hams
    parser.add_argument('--alg', type=str, default='hams', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--reward_death_value', type=int, default=80, help='reward value when each agent dead')
    parser.add_argument('--reward_win', type=int, default=400, help='reward for winning in an episode')
    parser.add_argument('--num_env_steps', type=int, default=10e7, help='Number of environment steps to train (default: 10e7)')
    # parser.add_argument('--episode_length', type=int, default=200, help='Steps of each episode, only for hams algorithm')
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 22001

    # the number of the episodes in one epoch
    args.n_episodes = 10

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 22001

    # the number of the episodes in one epoch
    args.n_episodes = 4

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size = 16  # original value: 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 22001

    # the number of the episodes in one epoch
    args.n_episodes = 4

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of ms-marl
def get_msmarl_args(args):
    # network
    args.master_hidden_dim = 128
    args.slave_hidden_dim = 64
    args.lr_master = 1e-3
    args.lr_slave = 1e-4

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = "epoch"

    # the number of the epoch to train the agent
    # args.n_epoch = 20001
    args.n_epoch = 22001

    # the number of the episodes in one epoch
    args.n_episodes = 4

    # the master index number in agents
    args.master_index = 0

    # the dimension for slave attention network
    args.attention_dim = 32

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of hams
def get_hams_args(args):
    # network
    args.master_hidden_dim = 128
    args.slave_hidden_dim = 64
    args.gcm_hidden_dim = 128
    args.critic_dim = 128
    args.lr_master = 1e-3  # master学习率 2s3z: 1e-2
    args.lr_slave = 1e-3  # slave学习率
    args.lr_comm = 1e-3  # commGroup学习率 2s3z: 1e-2
    args.lr_gcm = 1e-3  # gcm机制学习率
    args.lr_critic = 1e-2  # critic学习率

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = "epoch"

    # the number of the epoch to train the agent
    args.n_epoch = 22001  # 2s3z: 40501

    # the number of the episodes in one epoch 2s3z:4
    args.n_episodes = 8

    # the dimension for slave attention network
    args.attention_dim = 32

    # Decays the learning rate of each parameters group by gamma every step_size epochs
    args.lr_step_size = 500
    args.lr_gamma = 0.9

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # calculate the GAE
    args.tau = 0.95

    # PPO hyper-parameter
    args.ppo_epochs = 5  # 更新多少次
    args.num_mini_batch = 1  # 将batch分成多少个mini_batch
    args.clip_param = 0.2
    args.entropy_coef = 0.01
    args.value_loss_coef = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # Multiple process parameters
    args.n_rollout_threads = 4  # Number of parallel envs for training rollouts
    args.agent_scale = 4  # Multi-agent plan scale
    args.n_eval_rollout_threads = 1  # Number of parallel envs for evaluating rollouts
    args.top_k = 3

    # advantage noise ---add
    args.use_adv_noise = True  # whether to use the advantage noise
    args.alpha = 0.05

    # value noise ---add
    args.use_value_noise = True  # whether to use the value noise
    args.reset_interval = -1  # reset the noise every reset_interval epoch, -1 means infinity
    args.sigma = 1
    args.noise_dim = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
