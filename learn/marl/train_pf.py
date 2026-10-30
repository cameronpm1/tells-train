import os
import sys
import time
import yaml
import hydra
import torch
import shutil
import random
import logging
import pickle
import numpy as np
from torch import nn
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader, TensorDataset

from ray.tune.logger import pretty_print
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED,
)

#from logger import getlogger
from util.util import mkdir, load_config
from envs.marl.make_env import make_marl_env
from envs.marl.rllib_wrapper import RLLibWrapper
from controllers.predator_prey_control import compute_slot_actions
from controllers.market_game_control import compute_market_actions
from learn.marl.callbacks import CurriculumCallback, LogRawEpisodeReturn

try:
    from evals.belief.eval import load_model
except ImportError:
    print('Belief model loading not available, check virtual environment')


#logger = getlogger(__name__)

def _load_checkpoint_iteration(checkpoint_dir: str) -> int | None:
    state_path = os.path.join(checkpoint_dir, 'algorithm_state.pkl')
    if not os.path.exists(state_path):
        return None

    try:
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
    except Exception:
        return None

    training_iteration = state.get('training_iteration')
    if training_iteration is None:
        return None

    return int(training_iteration)

def _find_latest_checkpoint(logdir: str) -> tuple[str | None, int]:
    latest_dir = None
    latest_iter = -1

    logdir = os.path.abspath(logdir)

    if not os.path.exists(logdir):
        return None, latest_iter

    for entry in os.scandir(logdir):
        if not entry.is_dir():
            continue

        training_iteration = _load_checkpoint_iteration(entry.path)

        if training_iteration is None:
            continue

        if training_iteration > latest_iter:
            latest_iter = training_iteration
            latest_dir = entry.path

    return latest_dir, latest_iter

def _save_checkpoint(algo, checkpoint_dir: str):
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    algo.save(checkpoint_dir=checkpoint_dir)

def _build_algo(algo_config, logger_creator=None):
    try:
        return algo_config.build_algo(logger_creator=logger_creator)
    except TypeError as exc:
        if 'logger_creator' not in str(exc):
            raise
        print('RLlib build_algo does not accept logger_creator; using default logger.')
        return algo_config.build_algo()

def _episodes_done(terminations: dict, truncations: dict) -> bool:
    return all(bool(v) for v in terminations.values()) or all(bool(v) for v in truncations.values())

def _summarize_eval_rows(rows: list[dict]) -> dict[str, float]:
    metrics = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
    return {
        'success_rate': float(metrics['success'].mean()),
        'avg_steps': float(metrics['steps'].mean()),
    }

def _evaluate_shared_policy(algo, cfg: dict, runs: int = 20) -> dict[str, float]:
    policy_id = cfg['policy_list'][0]
    rows = []

    for seed_offset in range(runs):
        env = make_marl_env(cfg, seed=int(cfg['seed']) + seed_offset, wrap=None)
        obs, _ = env.reset()

        best_dist = float('inf')
        max_hold = 0
        success = False
        oob = False
        steps = 0

        for t in range(cfg['env']['max_episode_length']):
            action_dict = {'target': 0}
            for agent in cfg['env']['learned_agent_list']:
                action = algo.compute_single_action(obs[agent], policy_id=policy_id, explore=False)
                if isinstance(action, tuple):
                    action = action[0]
                action_dict[agent] = int(action)

            obs, _, terminations, truncations, infos = env.step(action_dict)

            info = infos['__common__'][cfg['env']['learned_agent_list'][0]]
            success = success or bool(info['success'])
            steps = t + 1

            if _episodes_done(terminations, truncations):
                break

        rows.append({
            'success': int(success),
            'steps': steps,
        })

    return _summarize_eval_rows(rows)

def _collect_controller_dataset(cfg: dict, episodes: int):
    obs_buf = []
    act_buf = []
    ret_buf = []
    gamma = float(cfg['alg']['gamma'])
    learned_agents = cfg['env']['learned_agent_list']

    for episode_idx in range(episodes):
        env = make_marl_env(
            cfg,
            seed=int(cfg['seed']) + episode_idx,
            wrap=None,
        )
        obs, infos = env.reset()
        trajectory = []

        for i in range(cfg['env']['max_episode_length']):


            if 'predator_prey' in cfg['env']['scenario']:
                expert_actions = compute_slot_actions(obs,env.unwrapped.obs_map)
            elif 'drone' in cfg['env']['scenario']:
                expert_actions = drone_controller(obs, env.unwrapped.obs_map, cfg['env'].get('controller_kwargs', {}))
            elif 'football' in cfg['env']['scenario']:
                expert_actions = compute_rondo_actions(obs,env.unwrapped.obs_map)
            elif 'fire' in cfg['env']['scenario']:
                expert_actions = extinguish_controller(infos['__common__']['decomposed_obs'])
            elif 'market' in cfg['env']['scenario']:
                expert_actions = compute_market_actions(obs, env.unwrapped.obs_map, env.unwrapped.controller_cfg)

            predator_obs = {
                agent: np.asarray(obs[agent], dtype=np.float32).copy()
                for agent in learned_agents
            }
            action_dict = {'target': 0, **expert_actions}
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            team_reward = float(rewards[learned_agents[0]])
            trajectory.append((predator_obs, expert_actions, team_reward))

            if _episodes_done(terminations, truncations):
                break
        
        env.close()

        discounted_returns = []
        ret = 0.0
        for _, _, reward in reversed(trajectory):
            ret = reward + gamma * ret
            discounted_returns.append(ret)
        discounted_returns.reverse()

        for (predator_obs, expert_actions, _), discounted_return in zip(trajectory, discounted_returns):
            for agent in learned_agents:
                obs_buf.append(predator_obs[agent])
                act_buf.append(int(expert_actions[agent]))
                ret_buf.append(float(discounted_return))

    return (
        np.stack(obs_buf),
        np.asarray(act_buf, dtype=np.int64),
        np.asarray(ret_buf, dtype=np.float32),
    )

def _maybe_pretrain_policy(algo, cfg: dict, logdir: str) -> bool:
    pretrain_cfg = cfg['alg'].get('pretrain', {})

    if not pretrain_cfg or not pretrain_cfg.get('enabled', False):
        return False

    if 'ppo' not in cfg['alg']['type']:
        print('Skipping expert pretrain because it is only implemented for PPO.')
        return False

    episodes = int(pretrain_cfg.get('episodes', 300))
    epochs = int(pretrain_cfg.get('epochs', 20))
    batch_size = int(pretrain_cfg.get('batch_size', 1024))
    learning_rate = float(pretrain_cfg.get('lr', 1e-3))
    value_loss_coef = float(pretrain_cfg.get('value_loss_coef', 0.05))
    grad_clip = float(pretrain_cfg.get('grad_clip', 1.0))
    eval_runs = int(pretrain_cfg.get('eval_runs', 20))
    policy_id = cfg['policy_list'][0]

    print(f'Collecting {episodes} expert episodes from the slot controller...')
    obs_arr, act_arr, ret_arr = _collect_controller_dataset(cfg, episodes=episodes)
    action_counts = {
        action: int((act_arr == action).sum())
        for action in range(int(algo.get_policy(policy_id).action_space.n))
    }
    print(
        'Expert dataset:',
        {
            'samples': int(obs_arr.shape[0]),
            'obs_dim': int(obs_arr.shape[1]),
            'action_counts': action_counts,
        },
    )

    policy = algo.get_policy(policy_id)
    model = policy.model
    device = next(model.parameters()).device
    model.train()

    class_counts = np.bincount(act_arr, minlength=int(policy.action_space.n))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()

    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    value_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.tensor(obs_arr, dtype=torch.float32),
        torch.tensor(act_arr, dtype=torch.long),
        torch.tensor(ret_arr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_ce = 0.0
        total_value = 0.0
        total_correct = 0.0
        total_samples = 0

        for obs_batch, act_batch, ret_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            ret_batch = ret_batch.to(device)

            logits, _ = model({'obs': obs_batch}, [], None)
            values = model.value_function()

            # This is straight behavior cloning on the policy head, with a light
            # value-head fit so PPO does not start from a completely useless critic.
            policy_loss = ce_loss(logits, act_batch)
            critic_loss = value_loss(values, ret_batch)
            loss = policy_loss + (value_loss_coef * critic_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_ce += float(policy_loss.item()) * len(obs_batch)
            total_value += float(critic_loss.item()) * len(obs_batch)
            total_correct += float((logits.argmax(dim=-1) == act_batch).float().sum().item())
            total_samples += len(obs_batch)

        print(
            f'Pretrain epoch {epoch + 1}/{epochs}:',
            {
                'policy_ce': total_ce / total_samples,
                'value_mse': total_value / total_samples,
                'accuracy': total_correct / total_samples,
            },
        )

    summary = _evaluate_shared_policy(algo, cfg, runs=eval_runs)
    print(f'Pretrain evaluation over {eval_runs} runs: {summary}')

    pretrain_dir = os.path.join(logdir, 'checkpoint0_pretrain')
    _save_checkpoint(algo, pretrain_dir)
    _save_checkpoint(algo, os.path.join(logdir, 'checkpoint_latest'))
    algo.restore(pretrain_dir)

    return True

def train(config_path: str, kwargs=None):

    cfg = load_config(config_path)
    logdir = os.path.abspath(cfg['logdir'])
    cfg['logdir'] = logdir
    if kwargs is not None:
        for key, value in kwargs.items():
            cfg[key] = value
    if not os.path.exists(logdir):
        print('Save directory not found, creating path ...')
        #logger.info("Save directory not found, creating path ...")
        mkdir(logdir)
    else:
        print('Save directory found ...')
        #logger.info("Save directory found ...")
    print("current directory:", logdir)
    #save copy of config file
    OmegaConf.save(config=cfg, f=logdir+'/config.yaml')

    def logger_creator(config):
        try:
            from ray.tune.logger import UnifiedLogger
            return UnifiedLogger(config, logdir, loggers=None)
        except:
            print('Cannot import unified logger, check ray version...')

    algo_config = make_ray_config(cfg)

    algo_build = _build_algo(algo_config, logger_creator=logger_creator)

    resume_dir, resume_iter = _find_latest_checkpoint(logdir)
    start_iter = 0

    if resume_dir is not None:
        print(f'Resuming from checkpoint: {resume_dir}')
        algo_build.restore(resume_dir)
        start_iter = max(resume_iter, 0)
        print(f'Restored training iteration: {start_iter}')
    else:
        initial_checkpoint = cfg['alg'].get('initial_checkpoint')
        if initial_checkpoint:
            initial_checkpoint = os.path.abspath(initial_checkpoint)
            print(f'Initializing from checkpoint: {initial_checkpoint}')
            algo_build.restore(initial_checkpoint)
        else:
            _maybe_pretrain_policy(algo_build, cfg, logdir)

    #train 15,000 iterations
    total_iters = int(cfg['alg']['timesteps'])
    checkpoint_freq = int(cfg['alg'].get('checkpoint_freq', 500))

    try:
        for i in range(start_iter, total_iters):
            result = algo_build.train()
            current_iter = i + 1

            if current_iter % checkpoint_freq == 0:
                save_dir = os.path.join(logdir, 'checkpoint' + str(current_iter))
                _save_checkpoint(algo_build, save_dir)
                _save_checkpoint(algo_build, os.path.join(logdir, 'checkpoint_latest'))
                print(pretty_print(result))
    except KeyboardInterrupt:
        interrupt_dir = os.path.join(logdir, 'checkpoint_latest')
        print(f'\nTraining interrupted. Saving latest checkpoint to: {interrupt_dir}')
        _save_checkpoint(algo_build, interrupt_dir)
        return

    save_dir = os.path.join(logdir, 'final')
    _save_checkpoint(algo_build, save_dir)


def make_ray_config(
    cfg: dict,
    wrapper: Optional[str] = 'pf',
):
    '''
    takes loaded config and returns ray config file for tarining rl

    input
    -----
    config:dict
        loaded config dictionary

    output
    ------
    ray_config class
    '''
    belief_kwargs = {}
    #make env function 
    def env_maker(config):

        seed = cfg['seed']

        if config is not None:
            seed += ((1000*config.worker_index + config.vector_index))   

            env = make_marl_env(cfg,seed=int(seed),wrap=wrapper,belief_kwargs=belief_kwargs)
        else:
            env = make_marl_env(cfg,seed=int(seed),wrap=wrapper)

        return env

        
    class policyTrainingSchedule():
        '''
        records how many batches have been used 
        '''
        def __init__(self,workers=1):
            self.workers=workers
            self.max_samples = 0

        def policy_training_schedule(self, pid, batch):
            '''
            determines when each policy should be trained
            '''
            return True
            
    policy_tracker = policyTrainingSchedule(workers=cfg['alg']['nenvs'])
    
    env_name = cfg['env']['scenario']
    register_env(env_name, env_maker) #register make env function
    #test_env for getting obs/action space
    test_env = env_maker(None)
    policy_list = cfg['policy_list']
    policy_mapping_fn = marl_policy_mapping_fn
    policy_training_fn = policy_tracker.policy_training_schedule

    batch = None

    training_kwargs = {}

    if 'sac' in cfg['alg']['type']:

        algo = SACConfig() 
        policy_model_dict = {
            'post_fcnet_hiddens': cfg['alg']['pi'],
        }
        q_model_dict = {
            'post_fcnet_hiddens': cfg['alg']['vf'],
        }
        batch = cfg['alg']['batch']*len(cfg['policy_list'])

        policy_info = {}

        for label in policy_list:

            policy_info[label] = PolicySpec(
                            None, #policy_class
                            test_env.get_observation_space(label), #observation_space
                            test_env.get_action_space(label), #action_space
                            {'lr':cfg['alg']['lr'],
                                'policy_model_config':policy_model_dict,
                                'q_model_config':q_model_dict,
                            } #config (gamma,lr,etc.)
                        )
        training_kwargs = {
            'gamma': cfg['alg']['gamma'],
            'train_batch_size': batch,
            'training_intensity': None,
            'grad_clip_by': 'norm',
            'replay_buffer_config': {
                'type': 'MultiAgentReplayBuffer',
                'capacity': 1000000,
                'replay_sequence_length': 1,
            },
        }
    elif 'ppo' in cfg['alg']['type']:
        algo = PPOConfig()
        model_dict = {
            'fcnet_hiddens': cfg['alg']['pi'],
        }
        batch = cfg['alg']['batch']

        policy_info = {}
        for label in policy_list:
            policy_info[label] = PolicySpec(
                None,
                test_env.get_observation_space(label),
                test_env.get_action_space(label),
                {},
            )

        training_kwargs = {
            'gamma': cfg['alg']['gamma'],
            'lr': cfg['alg']['lr'],
            'train_batch_size': batch,
            'model': model_dict,
            'lambda_': cfg['alg'].get('lambda', 0.95),
            'clip_param': cfg['alg'].get('clip_param', 0.2),
            'entropy_coeff': cfg['alg'].get('entropy_coeff', 0.01),
            'vf_clip_param': cfg['alg'].get('vf_clip_param', 10.0),
            'grad_clip': cfg['alg'].get('grad_clip', 0.5),
        }
    else:
        raise ValueError(f"Unsupported MARL algorithm type: {cfg['alg']['type']}")

    #initialize RL training algorithm using rllib's multi_agent settings
    algo_config = (algo
            .environment(env=env_name,
                        #env_config={'num_agents':3},
                        )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
            .env_runners(num_env_runners=cfg['alg']['nenvs'], #20
                        num_envs_per_env_runner=cfg['alg']['cpu_envs'], #60
                        num_cpus_per_env_runner=1,
                        #sample_timeout_s=120,
                        #rollout_fragment_length=256,
                        )
            .resources(num_gpus=cfg['alg'].get('num_gpus', 0))
            .multi_agent(policy_mapping_fn=policy_mapping_fn,
                            policies_to_train=policy_training_fn,
                            policies=policy_info)
            .training(**training_kwargs)
            .callbacks(callbacks_class=LogRawEpisodeReturn)
    )

    if 'ppo' in cfg['alg']['type']:
        algo_config.minibatch_size = cfg['alg'].get('minibatch_size', 512)
        algo_config.num_epochs = cfg['alg'].get('num_epochs', 10)
    
    del test_env

    return algo_config

def marl_policy_mapping_fn(agent_id, episode, **kwargs):

    return 'agent0' #for single policy, multiple agents

    #return agent_id
