"""One noise level of scripts/noise_sweep.sh: the noisy belief dataset, the belief training config and the evaluation
config, built with belief_noise_sweep.py's functions but written where the sweep script wants them, so nothing in
confs/ or the existing sweep datasets is touched:

    data/<source>_<tag>/                 noisy copy of the clean dataset data/<source>/ (same train/val/test split)
    <work>/belief_<tag>.yaml             belief training config; the model is trained into <work>/belief/
    <work>/eval_<tag>.yaml               evaluation config; its rollouts go to <model_dir>/results_allocentric_sweep_<tag>

Prints the three paths as KEY=value lines for the shell script.

    python evals/marl/noise_sweep_build.py --env market_game --sigma 0.3 --source noise_sweep_market_game_clean \
        --work logs/noise_sweep/market_game
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import evals.marl.belief_noise_sweep as B  # noqa: E402

# sweep env -> how its evaluation config adds the noise (see belief_noise_sweep.py):
#   adv    sigma on the belief input and on the observed adversary (+ market: 0.25 x sigma on the observed demand)
#   image  fire: each fire-window cell flipped to a random state with probability sigma
EVAL_KIND = {'predator_prey': 'adv', 'drones_xy': 'adv', 'market_game': 'adv', 'fire': 'image', 'football': 'adv'}
# per-env settings of the latest paper sweeps on top of belief_noise_sweep.py's configs
#   football (results/run_football_sweep_v3.py, 2026-11-06): sigma also on the ball position the policy receives
#   (belief_ball_noise), and the noisy belief models stop early after 500 epochs without improvement
EVAL_EXTRA = {'football': lambda sigma: {'belief_ball_noise': float(sigma)}}
BELIEF_EXTRA = {'football': {'early_stop_patience': 500}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=list(EVAL_KIND), required=True, help="belief_noise_sweep.py SWEEP_ENVS key")
    p.add_argument('--sigma', type=float, required=True)
    p.add_argument('--source', required=True, help='clean belief dataset under data/ (collected by the sweep script)')
    p.add_argument('--work', required=True, help='folder for the configs and the belief model')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--epochs', type=int, default=None, help='override the belief training epochs (testing)')
    p.add_argument('--num-workers', type=int, default=8, help='belief dataloader workers (several trainings share CPUs)')
    p.add_argument('--configs-only', action='store_true', help='write the configs without building the dataset')
    args = p.parse_args()

    E = dict(B.SWEEP_ENVS[args.env], source=args.source)
    tag = B.tag(args.sigma, E)
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)

    # noisy dataset (data/<source>_<tag>) with the source's split
    data_name = f"{E['source']}_{tag}" if args.configs_only else B.make_dataset(E, args.sigma, args.workers)

    # belief training config: as belief_noise_sweep.make_configs (the clean model's hyperparameters), but the model
    # is trained into the work folder
    cfg = yaml.safe_load(open(E['belief_config']))
    cfg.update(B.reference_hparams(E))
    cfg.update(data_name=data_name, label='noise', log_dir=os.path.join(work, 'belief') + '/',
               num_workers=args.num_workers)
    if E['epochs']:
        cfg['epochs'] = E['epochs']
    cfg.update(BELIEF_EXTRA.get(args.env, {}))
    if args.epochs:
        cfg['epochs'] = args.epochs
    belief_cfg = os.path.join(work, f'belief_{tag}.yaml')
    yaml.safe_dump(cfg, open(belief_cfg, 'w'), sort_keys=False)
    belief_dir = '_'.join([cfg['log_dir'], data_name, cfg['model_name'], 'noise']) + \
        '/lightning_logs/checkpoints/{epoch}_{val_loss}'

    # evaluation config: written by belief_noise_sweep into the work folder, with its own results suffix
    written = []
    B.config_path = lambda path, suffix: written.append(os.path.join(work, f'eval_{tag}.yaml')) or written[-1]
    if EVAL_KIND[args.env] == 'image':
        B.make_image_config(E, args.sigma)
    else:
        B.make_adv_config(E, args.sigma)
    eval_cfg = written[-1]
    ev = yaml.safe_load(open(eval_cfg))
    ev.update(EVAL_EXTRA.get(args.env, lambda sigma: {})(args.sigma))
    ev['results_suffix'] = f'_sweep_{tag}'
    yaml.safe_dump(ev, open(eval_cfg, 'w'), sort_keys=False)

    print(f'DATA_NAME={data_name}')
    print(f'BELIEF_CFG={belief_cfg}')
    print(f'BELIEF_DIR={belief_dir}')
    print(f'EVAL_CFG={eval_cfg}')
    print(f'RESULTS_SUFFIX=_sweep_{tag}')


if __name__ == '__main__':
    main()
