import multiprocessing
import os
import yaml
import numpy as np
from pathos.multiprocessing import ProcessPool  # type: ignore
from stable_baselines3 import PPO

from data_manager import Data
from environment import StockEnv


def run_single_env(day, save_directory, args):
    '''
    Run a single environment for evaluation
    @param j: What iteration of a specific environment is evaluated
    @param day: What day to evaluate
    @param args: Configuration arguments
    @return: Stats from how the agent acted
    '''

    # Define the path to the saved model
    model_path = os.path.join(os.path.join('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/runs/' + save_directory), 'agent.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the model within the worker
    model = PPO.load(model_path, device='cpu')

    # Initialize data and environment
    data = Data(args, False)
    file_name, test_files = data.load_test_file(day)


    env = StockEnv(test_files[0], test_files[1], True, args)
    state = env.reset()

    env_steps = env_reward = env_pos = 0
    profit_per_trade = []

    while True:
        env_steps += 1

        action, _ = model.predict(state)
        state, reward, done, obs = env.step(action)
        env_pos += obs['closed']
        env_reward += reward
        if obs['closed']:
            profit_per_trade += [[reward, obs['open_pos'], obs['closed_pos'], obs['position'], obs['action']]]

        if done:
            break

    return [[file_name, env_steps, env_pos, profit_per_trade, env_reward]], day


def eval_agent(args, save_directory):

    save_dir = os.path.join('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/run_rsults' + save_directory)
    os.makedirs(save_dir, exist_ok=True)

    # Define the path to the saved model and check its existence
    model_path = os.path.join('/Volumes/L96/DRL/DRL_for_Active_High_Frequency_Trading-main/runs/' + save_directory, 'agent.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Save the parameters for reference
    with open(os.path.join(save_dir, 'parameters.yaml'), 'w') as file:
        yaml.dump(vars(args), file)  # Changed to vars(args) for better compatibility

    n_test_files = len(os.listdir(os.path.join(os.getcwd(), args.data_dir, 'test_data')))
    jobs_to_run = args.eval_runs_per_env * n_test_files
    pool = ProcessPool(multiprocessing.cpu_count())

    # Prepare arguments without passing the model
    day_values = list(range(n_test_files))
    save_directory_values = [save_directory] * jobs_to_run
    args_values = [args] * jobs_to_run

    # Execute the evaluation in parallel
    for ret, n in pool.uimap(run_single_env, day_values, save_directory_values, args_values):
        eval_filename = os.path.join(save_dir, f'eval{n}.npy')
        np.save(eval_filename, np.array(ret, dtype=object), allow_pickle=True)
        print(f"Saved evaluation result to {eval_filename}")


