"""Belief-noise sweep: builds the noisy belief datasets and the configs for each environment and noise level.

For each noise std sigma (tag noise<sigma>):
    data/<source>_noise<sigma>/        copy of the clean belief dataset with Gaussian noise (std sigma) added to every
                                       belief INPUT value; labels (true teammate states) stay clean; same split
    confs/belief/<env>_noise<sigma>.yaml
                                       belief training config: the clean model's own hyperparameters (read from its
                                       checkpoint for every env but predator-prey) on that dataset
    <eval config>_noise<sigma>.yaml    (default) evaluation config: sigma on the belief's input at inference
    <eval config>_noise<sigma>_prey.yaml / _adv.yaml   (--adv) ALSO sigma on the adversary position the agent
                                       observes (env noise.target); the belief-input noise then skips the inputs
                                       that carry the observed adversary (found by perturbation, see
                                       adversary_inputs), so every input still carries exactly sigma, as in training
    fire (--latent): no datasets or training: sigma (in units of the latent's mean per-unit std) is added to the
                                       policy's conv latent at evaluation (env config latent_noise)

The rest of each sweep (training, evaluation, CCA, plot) is run by the results/run_noise_sweep*.sh scripts.

Usage:
    python evals/marl/belief_noise_sweep.py --sigmas 0.25 0.5 1.0                    # predator-prey datasets
    python evals/marl/belief_noise_sweep.py --env football --sigmas 0.04 0.075 0.1   # datasets + configs
    python evals/marl/belief_noise_sweep.py --env football --sigmas 0.04 --adv       # adversary-noise eval configs
    python evals/marl/belief_noise_sweep.py --env fire --sigmas 0.25 0.5 1 --latent  # latent-noise eval configs
"""
import argparse
import glob
import os
import shutil
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

SWEEP_ENVS = {
    'predator_prey': dict(
        source='predator_prey_fully_obs', belief_config='confs/belief/predator_prey.yaml', ref_belief=None,
        epochs=1000,  # the clean model's validation loss bottomed out at epoch 745 of 4000; the best checkpoint is kept
        eval_config='confs/predator_prey/3a_game_allocentric.yaml', adv_suffix='prey',
        packaging='predator_prey_obs_packaging', checkpoint='logs/marl/predator_prey_fully_obs/checkpoint1000',
        n_agents=3),
    'football': dict(  # 6000 episodes, but the clean model used only the stored 1000-episode split (ids 0-999).
        # (The 2026-11-03 sweep's datasets are named football_fully_obs_old_noise*: the source was renamed after.)
        source='football_fully_obs', belief_config='confs/belief/football.yaml',
        ref_belief='logs/belief/_football_fully_obs_football_NN', epochs=None,
        eval_config='confs/football/5a_game_allocentric.yaml', adv_suffix='adv',
        packaging='football_obs_packaging', checkpoint='logs/marl/football_fully_obs/checkpoint3000', n_agents=5),
    'drones': dict(  # the model's checkpoint calls its data drones_fully_obs_4: it is drones_fully_obs (same split)
        source='drones_fully_obs', belief_config='confs/belief/drones.yaml',
        ref_belief='logs/belief/_drones_fully_obs_drones_NN', epochs=None,
        eval_config='confs/drones/3a_game_allocentric.yaml', adv_suffix='adv',
        packaging='drones_obs_packaging', checkpoint='logs/marl/drones_fully_obs/checkpoint1500', n_agents=3),
    'market_game': dict(  # results_allocentric came from model _1, stopped at epoch 1466 (_2 was trained later)
        source='market_game_7a_fully_obs', belief_config='confs/belief/market_game.yaml',
        ref_belief='logs/belief/_market_game_7a_fully_obs_market_game_NN_1', epochs=1466,
        eval_config='confs/market_game/7a_game.yaml', adv_suffix='adv',
        packaging='market_game_obs_packaging', checkpoint='logs/marl/market_game_7a_fully_obs/checkpoint829',
        n_agents=7),
    # drones, horizontal only: drone altitude barely varies (std 0.12), so isotropic noise swamps it; here the noise
    # is on x and y only (belief inputs, training data and the observed intruder). Datasets/models tagged noisexy.
    'drones_xy': dict(
        source='drones_fully_obs', belief_config='confs/belief/drones.yaml',
        ref_belief='logs/belief/_drones_fully_obs_drones_NN', epochs=None, tag='noisexy',
        eval_config='confs/drones/3a_game_allocentric.yaml', adv_suffix='adv', exclude='altitude',
        target_scale=[1.0, 1.0, 0.0], packaging='drones_obs_packaging',
        checkpoint='logs/marl/drones_fully_obs/checkpoint1500', n_agents=3),
    # fire: noise in the IMAGE (cell flips), not Gaussian: the belief input is per-frame one-hot burning/burnt maps
    'fire': dict(
        source='fire_fully_obs', belief_config='confs/belief/fire.yaml', ref_belief='logs/belief/_fire_fully_obs_fire_NN',
        epochs=None, eval_config='confs/fire/4a_game_allocentric.yaml', noise_kind='flip', frames=4, cells=61 * 61),
}
HPARAMS = ['input_channels', 'output_channels', 'model_name', 'lr', 'train_batch', 'val_batch', 'test_batch',
           'lr_schedule', 'gamma', 'seed', 'epochs']


def tag(sigma, E=None):
    return f"{(E or {}).get('tag', 'noise')}{sigma:g}"


def config_path(path, suffix):
    root, ext = os.path.splitext(path)
    return f'{root}_{suffix}{ext}'


def flip_fire_input(x, p, rng, frames, cells):
    """Fire belief input = per frame [burning map, burnt map] (one-hot, cells each), then positions. Each cell of
    each frame is independently replaced, with probability p, by a uniformly random state (GREEN, RED, BLACK,
    WHITE): its burning / burnt bits become (state == RED) / (state == BLACK). Same noise as the evaluation's
    rllib_wrapper._fire_history_entry, which flips the fire window before it is one-hot encoded."""
    x = x.copy()
    for f in range(frames):
        burning = x[f * 2 * cells:(f * 2 + 1) * cells]
        burnt = x[(f * 2 + 1) * cells:(f * 2 + 2) * cells]
        flip = rng.random(cells) < p
        state = rng.integers(0, 4, size=int(flip.sum()))
        burning[flip] = state == 1
        burnt[flip] = state == 2
    return x


def noisy_episode(ep_dir, out_root, sigma, seed, flip=None, keep=None):
    """Copy one episode folder, adding noise to each sample's input (seeded per episode, reproducible): Gaussian
    with std sigma, or (flip = dict(frames, cells)) fire cell flips with probability sigma."""
    rng = np.random.default_rng([seed, int(os.path.basename(ep_dir))])
    out_dir = os.path.join(out_root, os.path.basename(ep_dir))
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(ep_dir)):
        d = np.load(os.path.join(ep_dir, f))
        x = d['input']
        eps = sigma * rng.standard_normal(x.shape)
        if keep is not None:  # inputs that get no noise (e.g. drone altitude)
            eps[keep] = 0.0
        noisy = flip_fire_input(x, sigma, rng, **flip) if flip else (x + eps).astype(x.dtype)
        np.savez(os.path.join(out_dir, f), input=noisy, label=d['label'])
    return len(os.listdir(ep_dir))


def make_dataset(E, sigma, workers, seed=0):
    source = E['source']
    name = f'{source}_{tag(sigma, E)}'
    out_root = os.path.join('data', name)
    eps = sorted(glob.glob(os.path.join('data', source, '*')))
    if os.path.isdir(out_root) and len(os.listdir(out_root)) == len(eps):
        print(f'{name}: already built', flush=True)
    else:
        with Pool(workers) as p:
            flip = dict(frames=E['frames'], cells=E['cells']) if E.get('noise_kind') == 'flip' else None
            keep = excluded_inputs(E) or None
            n = sum(p.map(partial(noisy_episode, out_root=out_root, sigma=sigma, seed=seed, flip=flip, keep=keep), eps))
        print(f'{name}: {len(eps)} episodes, {n} samples', flush=True)
    src_split = os.path.join('data', 'datainfo', source)
    if not glob.glob(os.path.join(src_split, '*.json')):  # no stored split for the source: make it once, then share it
        from learn.belief.preprocess import split_data
        split_data('data/', name=source, seed=reference_hparams(E).get('seed', 2))
    split_dir = os.path.join('data', 'datainfo', name)
    os.makedirs(split_dir, exist_ok=True)
    for f in glob.glob(os.path.join(src_split, '*.json')):  # identical train/val/test split
        shutil.copy(f, split_dir)
    return name


def reference_hparams(E):
    """Hyperparameters the clean belief model was actually trained with (its checkpoint), where it is known."""
    if not E['ref_belief']:
        return {}
    import torch
    ckpt = glob.glob(os.path.join(E['ref_belief'], 'lightning_logs', 'checkpoints', '*', '*.ckpt'))[0]
    h = torch.load(ckpt, map_location='cpu', weights_only=False)['hyper_parameters']
    return {k: h[k] for k in HPARAMS if k in h}


def make_configs(env, E, sigma, data_name):
    cfg = yaml.safe_load(open(E['belief_config']))
    cfg.update(reference_hparams(E))
    cfg['data_name'] = data_name
    if E['ref_belief']:  # predator-prey keeps its original label (its sweep scripts expect those model folders)
        cfg['label'] = 'noise'
    if E['epochs']:
        cfg['epochs'] = E['epochs']
    belief_path = f'confs/belief/{env}_{tag(sigma, E)}.yaml'
    yaml.safe_dump(cfg, open(belief_path, 'w'), sort_keys=False)
    if E.get('noise_kind') == 'flip' or E.get('exclude'):  # their evaluation configs come from --image / --adv
        print(f'config: {belief_path}', flush=True)
        return
    ev = yaml.safe_load(open(E['eval_config']))
    ev.update(belief_input_noise=float(sigma), results_suffix=f'_{tag(sigma)}')
    eval_path = config_path(E['eval_config'], tag(sigma))
    yaml.safe_dump(ev, open(eval_path, 'w'), sort_keys=False)
    print(f'configs: {belief_path}, {eval_path}', flush=True)


def adversary_inputs(E):
    """Belief inputs that carry the observed adversary position: perturb target_pos in every step of a real
    observation history and see which packaged inputs move."""
    import copy
    import evals.marl.observation_packaging as P
    pack = getattr(P, E['packaging'])
    d = np.load(os.path.join(E['checkpoint'], 'results', '0.npz'), allow_pickle=True)
    om = d['obs_map'].item()
    agents = [f'agent{i}' for i in range(E['n_agents'])]
    hist = [{a: np.asarray(d[str(k)][0][a], dtype=float) for a in agents} for k in range(1, 13)]
    base = pack(hist, om, ['agent0'], min_obs=10)[0]['agent0']['input']
    moved = np.zeros(len(base), dtype=bool)
    for step in range(len(hist)):  # one step at a time, so no perturbations cancel
        h2 = copy.deepcopy(hist)
        h2[step]['agent0'][om['target_pos']] += 1.0
        moved |= np.abs(pack(h2, om, ['agent0'], min_obs=10)[0]['agent0']['input'] - base) > 1e-9
    return np.flatnonzero(moved).tolist()


def excluded_inputs(E):
    """Belief inputs that never get noise: for exclude='altitude', every input that moves when the altitude (z) of
    the agent, the intruder or a teammate is perturbed."""
    if E.get('exclude') != 'altitude':
        return []
    import copy
    import evals.marl.observation_packaging as P
    pack = getattr(P, E['packaging'])
    d = np.load(os.path.join(E['checkpoint'], 'results', '0.npz'), allow_pickle=True)
    om = d['obs_map'].item()
    agents = [f'agent{i}' for i in range(E['n_agents'])]
    hist = [{a: np.asarray(d[str(k)][0][a], dtype=float) for a in agents} for k in range(1, 13)]
    base = pack(hist, om, ['agent0'], min_obs=10)[0]['agent0']['input']
    moved = np.zeros(len(base), dtype=bool)
    for key in ('self_pos', 'target_pos', 'team'):
        idx = np.arange(om[key].start, om[key].stop)[2::3]  # z of each 3-d position in that block
        for step in range(len(hist)):
            for j in idx:
                h2 = copy.deepcopy(hist)
                h2[step]['agent0'][j] += 1.0
                moved |= np.abs(pack(h2, om, ['agent0'], min_obs=10)[0]['agent0']['input'] - base) > 1e-9
    return np.flatnonzero(moved).tolist()


def make_adv_config(E, sigma):
    """Evaluation config that ALSO puts sigma on the adversary position the agent observes (env noise.target): the
    policy sees a noisy adversary, and so does the belief through its history. The belief-input noise then skips
    those inputs, so every input still carries exactly sigma, as in training (no retraining needed)."""
    skip = sorted(set(adversary_inputs(E)) | set(excluded_inputs(E)))
    ev = yaml.safe_load(open(E['eval_config']))
    scale = E.get('target_scale')  # per-axis noise on the observed adversary (drones_xy: x, y only)
    ev['env']['noise']['target'] = [float(sigma) * a for a in scale] if scale else float(sigma)
    ev.update(belief_input_noise=float(sigma), belief_input_noise_skip=skip,
              results_suffix=f"_{tag(sigma, E)}_{E['adv_suffix']}")
    path = config_path(E['eval_config'], f"{tag(sigma, E)}_{E['adv_suffix']}")
    yaml.safe_dump(ev, open(path, 'w'), sort_keys=False)
    print(f'config: {path} (belief noise skips {len(skip)} adversary inputs: {skip})', flush=True)


def make_image_config(E, p):
    """Fire: each fire-window cell flipped to a random state with probability p, for the belief's history and the
    policy's image (rllib_wrapper._fire_history_entry), at evaluation."""
    ev = yaml.safe_load(open(E['eval_config']))
    ev.update(fire_image_noise=float(p), results_suffix=f'_image{p:g}')
    path = config_path(E['eval_config'], f'image{p:g}')
    yaml.safe_dump(ev, open(path, 'w'), sort_keys=False)
    print(f'config: {path}', flush=True)


def make_latent_config(E, sigma, scale):
    """Fire: sigma x scale (the latent's mean per-unit std on fully-observed data) added to the policy's conv
    latent at evaluation."""
    ev = yaml.safe_load(open(E['eval_config']))
    ev.update(latent_noise=float(sigma) * scale, results_suffix=f'_latent{sigma:g}')
    path = config_path(E['eval_config'], f'latent{sigma:g}')
    yaml.safe_dump(ev, open(path, 'w'), sort_keys=False)
    print(f'config: {path}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=list(SWEEP_ENVS), default='predator_prey')
    p.add_argument('--sigmas', type=float, nargs='+', required=True)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--adv', '--prey', dest='adv', action='store_true',
                   help='only write the adversary-noise evaluation configs')
    p.add_argument('--latent', action='store_true', help='fire: only write the latent-noise evaluation configs')
    p.add_argument('--image', action='store_true', help='fire: only write the image-noise evaluation configs')
    args = p.parse_args()
    E = SWEEP_ENVS[args.env]
    if args.image:
        for sigma in args.sigmas:
            make_image_config(E, sigma)
        return
    if args.latent:
        import evals.marl.cca_analysis as A
        scale = A.fire_latent_scale(args.workers)
        print(f'latent scale (mean per-unit std, fully observed): {scale:.4f}', flush=True)
        for sigma in args.sigmas:
            make_latent_config(E, sigma, scale)
        return
    if args.adv:
        for sigma in args.sigmas:
            make_adv_config(E, sigma)
        return
    for sigma in args.sigmas:
        make_configs(args.env, E, sigma, make_dataset(E, sigma, args.workers))
    print('DATASETS DONE', flush=True)


if __name__ == '__main__':
    main()
