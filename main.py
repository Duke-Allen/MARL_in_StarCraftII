from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import *
import os
from envs.create_env import *
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()

        # replay directory
        if args.replay_dir != '':
            args.replay_dir = args.replay_dir + '/' + args.alg + '/' + args.map
            if not os.path.exists(args.replay_dir):
                os.makedirs(args.replay_dir)

        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        elif args.alg.find('msmarl') > -1:
            args = get_msmarl_args(args)
        elif args.alg.find('hams') > -1:
            args = get_hams_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)

        # Environment
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir,
                            # reward_only_positive=False,
                            reward_death_value=args.reward_death_value,
                            reward_win=args.reward_win,
                            )
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        # Initialize algorithm
        if args.alg != 'hams':
            runner = Runner(env, args)
        else:
            from hams_runner import HAMSRunner
            envs = make_train_env(args)  # train env
            eval_envs = make_eval_env(args)  # evaluate env
            args.envs = envs
            args.eval_envs = eval_envs
            runner = HAMSRunner(env, args)  # Use multiprocess for hams

        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break

        if args.alg != 'hams':
            env.close()
        else:
            envs.close()
