from smac.env import StarCraft2Env
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

"""Create the multiprocess environment for training an evaluation."""


# Build more env in multiprocess to collect data
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            environment = StarCraft2Env(map_name=all_args.map,
                                        step_mul=all_args.step_mul,
                                        difficulty=all_args.difficulty,
                                        game_version=all_args.game_version,
                                        seed=all_args.seed + rank * 1000,
                                        replay_dir=all_args.replay_dir,
                                        reward_death_value=all_args.reward_death_value,
                                        reward_win=all_args.reward_win)
            return environment

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


# Environment for evaluation
def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            eval_env = StarCraft2Env(map_name=all_args.map,
                                     step_mul=all_args.step_mul,
                                     difficulty=all_args.difficulty,
                                     game_version=all_args.game_version,
                                     seed=all_args.seed * 50000 + rank * 10000,
                                     replay_dir=all_args.replay_dir,
                                     reward_death_value=all_args.reward_death_value,
                                     reward_win=all_args.reward_win)
            return eval_env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
