import os
import json
import argparse


def repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def default_ray_base_dir():
    for env_name in ('SLURM_TMPDIR', 'SCRATCH', 'TMPDIR'):
        value = os.environ.get(env_name)
        if value:
            return os.path.join(value, 'tells_ray')
    return os.path.join(repo_root(), '.ray')


def set_torch_threads(args):
    import torch

    threads = args.torch_threads
    if threads is None:
        threads = 9
    torch.set_num_threads(threads)
    return threads


def init_ray(args):
    import ray

    if ray.is_initialized():
        return

    ray_base = os.path.abspath(args.ray_base_dir or default_ray_base_dir())
    ray_temp_dir = os.path.abspath(args.ray_temp_dir or os.path.join(ray_base, 'tmp'))
    ray_spill_dir = os.path.abspath(args.ray_spill_dir or os.path.join(ray_base, 'spill'))
    ray_working_dir = os.path.abspath(args.ray_working_dir or repo_root())

    os.makedirs(ray_temp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)

    runtime_env = None
    if not args.no_ray_runtime_env:
        runtime_env = {
            'working_dir': ray_working_dir,
            'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'},
            'excludes': ['.git/', 'logs/', 'data/', 'lab-logs/', '.ray/'],
        }

    ray_kwargs = {
        'num_gpus': args.ray_gpus,
        'include_dashboard': args.ray_dashboard,
        'runtime_env': runtime_env,
        '_temp_dir': ray_temp_dir,
        '_system_config': {
            'object_spilling_config': json.dumps({
                'type': 'filesystem',
                'params': {'directory_path': ray_spill_dir},
            }),
        },
    }
    if args.ray_cpus is not None:
        ray_kwargs['num_cpus'] = args.ray_cpus

    print('Initializing Ray:', {
        'working_dir': ray_working_dir,
        'temp_dir': ray_temp_dir,
        'spill_dir': ray_spill_dir,
        'num_cpus': ray_kwargs.get('num_cpus', 'auto'),
        'num_gpus': args.ray_gpus,
    })
    ray.init(**ray_kwargs)


def configure_marl_runtime(args):
    import warnings

    os.environ.setdefault('XDG_RUNTIME_DIR', '/tmp')
    os.environ.setdefault('RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO', '0')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='pygame')

    threads = set_torch_threads(args)
    init_ray(args)
    return threads


def add_runtime_args(parser):
    parser.add_argument('--torch_threads', type=int, default=None, help='Torch intra-op threads. Defaults to a conservative CPU fraction.')
    parser.add_argument('--ray_cpus', type=int, default=None, help='Total CPUs exposed to Ray. Defaults to Ray auto-detection.')
    parser.add_argument('--ray_gpus', type=float, default=0.0, help='Total GPUs exposed to Ray.')
    parser.add_argument('--policy_gpus', type=float, default=None, help='GPUs assigned to the RLlib algorithm config.')
    parser.add_argument('--ray_base_dir', type=str, default=None, help='Base directory for Ray temp and spill files.')
    parser.add_argument('--ray_temp_dir', type=str, default=None, help='Ray temp directory.')
    parser.add_argument('--ray_spill_dir', type=str, default=None, help='Ray object spill directory.')
    parser.add_argument('--ray_working_dir', type=str, default=None, help='Directory shipped to Ray workers. Defaults to this repo.')
    parser.add_argument('--ray_dashboard', action='store_true', help='Enable the Ray dashboard.')
    parser.add_argument('--no_ray_runtime_env', action='store_true', help='Do not package a Ray runtime_env working_dir.')
    parser.add_argument('--logdir', type=str, default=None, help='Override the config logdir for a new training/eval output directory.')
    parser.add_argument('--initial_checkpoint', type=str, default=None, help='Initialize MARL training from an existing RLlib checkpoint when the logdir has no checkpoint.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default='eval_rl', help='Command to run')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs to test')
    parser.add_argument('--model_dir', type=str, default=None, help='Model directory for evaluation')
    parser.add_argument('--belief_config', type=str, default=None, help='Config file for belief model')
    parser.add_argument('--belief_dir', type=str, default=None, help='Model directory for belief model')
    parser.add_argument('--belief_lr', type=float, default=None, help='Override the config lr when continuing belief training from --belief_dir')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save evaluation data to')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers to use for data collection')
    parser.add_argument('--no_eval_videos', action='store_true', help='Skip GIF rendering during MARL eval.')
    parser.add_argument('--collect_results', action='store_true', help='Store info from eval runs in a results folder.')
    add_runtime_args(parser)
    args = parser.parse_args() 

    if args.config is None:
        print('ERROR: No config file provided')

    elif args.command == 'marl_train':

        configure_marl_runtime(args)
        from learn.marl.train import train

        print('Training marl policies with config:', args.config)
        train(args.config)

    elif  args.command == 'marl_eval':

        configure_marl_runtime(args)
        from evals.marl.eval import eval

        print('Evaluating RL model with config:', args.config)
        print('Loading model from:', args.model_dir)
        eval(args.config,args.model_dir,args.runs, save_videos=not args.no_eval_videos, collect_results=args.collect_results,
             n_workers=args.n_workers)

    elif args.command == 'marl_eval_belief':

        configure_marl_runtime(args)
        from evals.marl.eval import eval

        print('Evaluating RL model with config:', args.config)
        print('Loading model from:', args.model_dir)
        eval(
            args.config,
            args.model_dir,
            args.runs,
            args.belief_dir,
            args.belief_config,
            save_videos=not args.no_eval_videos,
            collect_results=args.collect_results,
            n_workers=args.n_workers,
        )

    elif args.command == 'marl_train_belief':

        configure_marl_runtime(args)
        from learn.marl.train import train

        print('Training RL model with config:', args.config)
        print('Loading model from:', args.model_dir)

        kwargs = {
            'belief_config_dir': args.belief_config,
            'belief_dir' : args.belief_dir
        }

        train(args.config,kwargs=kwargs)

    elif 'marl_collect_data' == args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:

            set_torch_threads(args)
            from evals.marl.collect_eval_data import collect_data

            if args.save_dir is None:
                save_dir = 'data/' + args.model_dir.split('rl/')[1].split('/')[0]
            else:
                save_dir = 'data/' + args.save_dir
            print('Collecting data with RL model and config:', args.config)
            print('Loading model from:', args.model_dir)
            print('Saving data to:', save_dir)
            
            collect_data(
                args.config,
                args.model_dir,
                save_dir,
                n_runs=args.runs,
                n_workers=args.n_workers,
            )

    elif 'belief_train' in args.command:

        from learn.belief.train import train

        print('Training belief model with config:', args.config)
        train(args.config, belief_dir=args.belief_dir, lr=args.belief_lr)

    elif 'belief_eval' in args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:
        
            from evals.belief.eval import eval
            
            print('Evaluating belief model with config:', args.config)
            print('Loading model from:', args.model_dir)

            eval(args.config,args.model_dir)

    elif 'ego_eval' == args.command:

        configure_marl_runtime(args)
        from evals.marl.eval_pf import eval

        print('Evaluating PF model with config:', args.config)
        print('Loading model from:', args.model_dir)
        eval(
            args.config,
            args.model_dir,
            args.runs,
            save_videos=not args.no_eval_videos,
            collect_results=args.collect_results,
            n_workers=args.n_workers,
        )

    elif 'ego_train' == args.command:

        configure_marl_runtime(args)
        from learn.marl.train_pf import train

        print('Training PF model with config:', args.config)
        train(args.config,{})
