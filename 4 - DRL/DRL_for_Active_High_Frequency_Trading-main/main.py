import argparse
import time
import os
import random
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from data_manager import *
from environment import StockEnv
from eval_environment import eval_agent


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    if args.seed is None:
        args.seed = int(random.random() * 10000)

    save_directory = str(time.time()) + (args.tag if args.tag is not None else '')


    data = Data(args, True)
    print(len(data.snapshots))

    envs = [StockEnv(snapshots[0], snapshots[1], False, args)
            for snapshots in data.snapshots]
    print(len(envs))
    for idx, (scaled, unscaled) in enumerate(data.snapshots):
        # Check if each snapshot has at least last_n_ticks rows
        if scaled.shape[0] < args.last_n_ticks:
            print(f"Warning: Snapshot {idx} has only {scaled.shape[0]} rows, needs at least {args.last_n_ticks}")

    train_agent(envs, save_directory)
    eval_agent(args, save_directory)
    print("Evaluation finished! Results saved to runs_results/" + save_directory)



def train_agent(envs, save_directory):
    os.makedirs('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/runs/' + save_directory, exist_ok=False)

    with open(os.path.join('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/runs/' + save_directory, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    model = PPO('MultiInputPolicy', verbose=1, env=envs[0],
                gamma=args.gamma, ent_coef=args.ent_coef, max_grad_norm=args.grad_clip,
                learning_rate=args.lr, policy_kwargs=dict(net_arch=args.net_arch), seed=args.seed,
                device='cpu', batch_size=512)

    for i in range(len(envs)):
        logger = configure(os.path.join('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/runs/' + save_directory, str(i)), ["csv", "tensorboard"])
        model.set_logger(logger)
        model.set_env(envs[i])
        model.learn(args.epochs * args.snapshot_size, reset_num_timesteps=False)
        envs[i].close()
        print(f"Finished training on environment {i+1}/{len(envs)}")
    model.save(os.path.join(os.path.join('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/runs/' + save_directory), 'agent'))
    print("Training finished! Model saved successfully.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--last_n_ticks", type=int, default=10,
                        help='How many ticks to include in the state (Current tick + last_n_ticks - 1 past ticks)')
    parser.add_argument("--snapshot_size", type=int, default=5000,
                        help='Size of snapshots the agent will be trained on')
    parser.add_argument("--snapshots_per_day", type=int, default=3,
                        help='How many snapshots to create per training day')
    parser.add_argument("--tot_snapshots", type=int, default=25, help='How many snapshots to sample from all snapshots')
    parser.add_argument("--start_end_clip", type=int, default=int(0),
                        help='How many ticks to remove from the start and end of each day')
    parser.add_argument("--epochs", type=int, default=30, help='Epochs to train the agent on')
    parser.add_argument("--lr", type=float, default=.00032)
    parser.add_argument("--grad_clip", type=float, default=.5)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--ent_coef", type=float, default=.002)
    parser.add_argument("--net_arch", default=[64, 64], type=int)

    parser.add_argument("--data_dir", type=str, default='/Volumes/L96/DRL/days/', help='Directory where the data is stored')
    parser.add_argument("--rescale", type=str2bool, default=True,
                        help='Set to True if you want to rescale the data (ie. If you change the training data)')
    parser.add_argument("--resample", type=str2bool, default=False,
                        help='Set to True if you want to resample a set of snapshots from the training data')
    parser.add_argument("--use_m_t_m", type=str2bool, default=True,
                        help='Whether to use market to market value as a feature')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help='Tag to add to the save directory')

    parser.add_argument("--eval_runs_per_env", type=int, default=1)

    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)

    main(args)
