import inspect
import functools

import numpy as np
import torch


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def td_lambda_target(batch, max_episode_len, q_targets, args):
    # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
    # q_targets.shape = (episode_num, max_episode_len， n_agents)
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    r = batch['r'].repeat((1, 1, args.n_agents))
    # --------------------------------------------------n_step_return---------------------------------------------------
    '''
    1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
    最后一维,第n个数代表 n+1 step。
    2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
    否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
    如果没有置0，在计算td-error后再置0是来不及的
    3. terminated用来将超出当前episode长度的q_targets和r置为0
    '''
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # 最后计算1 step return
        n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
        # 同时要注意n step return对应的index为n-1
        for n in range(1, max_episode_len - transition_idx):
            # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
            # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
            n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
    # --------------------------------------------------n_step_return---------------------------------------------------

    # --------------------------------------------------lambda return---------------------------------------------------
    '''
    lambda_return.shape = (episode_num, max_episode_len，n_agents)
    '''
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    # --------------------------------------------------lambda return---------------------------------------------------
    return lambda_return


def compute_gae(next_value, values, rewards, padded, args):
    values = values + [next_value]
    gae = 0
    returns = []
    mask = (1 - np.array(padded)).tolist()
    for step in reversed(range(len(rewards))):
        # r = rewards[step][0].reshape([-1, 1])
        # v1 = args.gamma * values[step + 1][:, 0] * mask[step][0]
        # v = values[step][:, 0]
        # tmp = r + v1 - v
        delta = rewards[step] + args.gamma * values[step + 1][:, 0] * mask[step][0][0] - values[step][:, 0]  # (n_threads, 1)
        gae = delta + args.gamma * args.tau * mask[step][0][0] * gae
        returns.insert(0, np.array(gae + values[step][:, 0])[:, np.newaxis, :].repeat(args.n_agents, axis=-2))
    return returns


def merge_batch(state, obs, last_actions, avail_actions):
    """
    在采样过程中将一个时间步对应的信息合成一个样本，方便送入模型前向进行计算。
    Input：
    state: type is np.Array, dim is (n_threads, state_shape).
    obs: type is np.Array, dim is (n_threads, n_agents, obs_shape).
    last_actions: type is np.Array, dim is (n_threads, n_agents, n_actions).
    avail_actions: type is np.Array, dim is (n_threads, n_agents, n_actions).
    n_slaves: the slave agent number in each cluster.

    Output:
    episode：采集到的信息，每个元素的维度(n_threads, n_agents, shape)
    """
    # t_obs = np.expand_dims(obs, axis=1)  # (n_threads, episode_len, n_agents, obs_shape)
    t_obs = obs
    n_agents = t_obs.shape[-2]

    t_state = np.expand_dims(state, axis=1).repeat(n_agents, axis=1)  # (n_threads, n_agents, state_shape)
    # t_state = np.expand_dims(t_state, axis=1)  # (n_threads, episode_len, n_agents, state_shape)

    t_last_actions = last_actions
    t_avail_actions = avail_actions
    # t_last_actions = np.expand_dims(last_actions, axis=1)  # (n_threads, episode_len, n_agents, n_actions)
    # t_avail_actions = np.expand_dims(avail_actions, axis=1)  # (n_threads, episode_len, n_agents, n_actions)

    episode = dict(s=t_state.copy(),
                   o=t_obs.copy(),
                   last_u_onehot=t_last_actions.copy(),
                   avail_u=t_avail_actions.copy(),
                   )

    return episode


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


def check(inputs):
    """
    Transform numpy array to tensor.
    """
    output = torch.from_numpy(inputs) if type(inputs) == np.ndarray else inputs
    return output


def tensor2numpy(inputs):
    """
    Transform tensor to numpy array.
    """
    output = inputs.cpu().detach().numpy() if torch.is_tensor(inputs) else inputs
    return output
