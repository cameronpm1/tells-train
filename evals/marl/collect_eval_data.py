import os
import cv2
import yaml
import copy
import shutil
import numpy as np
from gymnasium import spaces
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config, marl_policy_mapping_fn

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def collect_data(
        config_dir:str,
        checkpoint_dir:str,
        master_dir:str,
        n_runs:int=1,
        seed:int=0,
        n_workers:int=1,
    ):
    '''
    test policy and save gif of test run

    input
    -----
    config_dir:str
        config directory
    model_dir:str
        policy directory
    master_dir:str
        directory to save gif
    n_runs:int
        number of test runs
    n_workers:int
        number of parallel workers to use for data collection
    '''
    mkdir(master_dir)

    cfg = load_config(config_dir)
    target_obs = cfg['min_obs']
    cfg['min_obs'] = cfg['horizon'] * 10

    run_segments = n_runs // n_workers
    run_segments_remainder = n_runs % n_workers

    args_list = []
    offset = 0

    for i in range(n_workers):
        if i == 0:
            n_runs = run_segments + run_segments_remainder
        else:
            n_runs = run_segments
        args_list.append(
            (
                config_dir,
                checkpoint_dir,
                master_dir,
                cfg['seed'] + i,
                n_runs,
                offset,
            )
        )
        offset += n_runs

    with Pool(n_workers) as pool:
        results = pool.starmap(data_worker, args_list)

    
def data_worker(
        config_dir:dict,
        checkpoint_dir:str,
        master_dir:str,
        seed:int,
        n_runs:int=1,
        dir_offset:int=0,
    ):
    '''
    make env and test policy for n_runs

    input
    -----
    config_dir:str
        config directory
    model_dir:str
        policy directory
    master_dir:str
        directory to save gif
    n_runs:int
        number of test runs
    dir_offset:int
        number of to offset save folders by (for parallel data collection)
    '''

    cfg = load_config(config_dir)
    min_obs = cfg['min_obs']
    cfg['min_obs'] = cfg['horizon'] * 10
    cfg['seed'] = seed 
    env = make_marl_env(
        cfg,
        seed=cfg['seed'],
        wrap='rllib',
        eval=True
    )
    obs_map = getattr(env.env, 'obs_map', 1)

    if checkpoint_dir is not None:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        # each worker only runs the policy locally: a small Ray instance (by default Ray sizes itself to every CPU
        # and pre-starts a worker process per CPU, ~200 processes per collection worker) and no env runners,
        # as in evals/marl/eval.py
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
        algo_config = make_ray_config({**cfg, 'alg': {**cfg['alg'], 'nenvs': 0}})
        algo = algo_config.build_algo()
        algo.restore(checkpoint_dir)
    else:
        print('No model directory provided...')
        exit()

    policy_list = cfg['policy_list']
    policy_mapping_fn = marl_policy_mapping_fn

    for n in range(n_runs):

        save_dir = os.path.join(master_dir,str(n+dir_offset))
        mkdir(save_dir)

        episode_agent = np.random.choice(cfg['env']['learned_agent_list'])
        target_obs = {}
        target_obs[episode_agent] = []

        obs, infos = env.reset()
        done = {"__all__": False}
        step = 0
        obs_history = []
        if 'fire' in env.name:
            obs_history.append(infos['__common__']['decomposed_obs'])
        else:
            obs_history.append(obs)

        #for agent_name in cfg['env']['learned_agent_list']:

        while not done["__all__"]:

            actions = {}

            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(
                    agent_obs,
                    policy_id=policy_mapping_fn(agent_id, 0), 
                    explore=False
                )
                actions[agent_id] = action
            step += 1

            obs, rewards, terminations, truncations, infos = env.step(actions)
            if 'fire' in env.name:
                obs_history.append(infos['__common__']['decomposed_obs'])
            else:
                obs_history.append(obs)

            done = {
                "__all__": terminations["__all__"] or truncations["__all__"]
            }

            data_point, _ = env.obs_packaging_func(obs_history, obs_map, [episode_agent], min_obs=min_obs, noise=env.temp_noise)
            save_path = os.path.join(save_dir,'step_'+str(step)+'_'+episode_agent+'.npz')
            np.savez(save_path,**data_point[episode_agent])


