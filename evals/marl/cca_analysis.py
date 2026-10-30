"""Fixed-axis CCA comparison of belief models against the true state (see CCA_ANALYSIS.md).

Every method (fully observed, allocentric, PF) is scored on the SAME axes, fixed by the true model:
    task axes       the policy's task directions: CCA of the true state with the fully-observed RL
                    policy's actions on the fully-observed data (sigma^2 > --min-eig), own state
                    regressed out, applied to every method as the same linear functionals of the state
    physical axes   teammate centroid, teammate spread, adversary and any extra entity (football: ball)

Per axis and method, raw R^2 = 1 - MSE / var(truth) of the belief's own estimate (no fitting) - exactly
what the policy receives. Coverage = mean R^2 over the task directions. CIs from an episode bootstrap.

Variables per agent-step (all agents pooled, first --max-steps steps of each episode):
    z      true teammates, adversary and extras relative to the agent (obs_no_noise)
    z_hat  belief teammates (sampled_predictions), observed adversary and extras (PF: filter estimates)
    own    the agent's own true state, regressed out of the task CCA
    y      every agent's action from the decentralized fully-observed RL policy on its true (noise-free)
           observation, one-hot (policy_actions_true_obs, logged by eval.py)

Usage:
    python evals/marl/cca_analysis.py --env football --out-dir results/football/cca_final
    python plotting/plot_cca_analysis.py --in-dir results/football/cca_final
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '24')

import argparse
import glob
import json
from functools import partial
from multiprocessing import Pool

import numpy as np

METHODS = ['fully_obs', 'allocentric', 'pf']  # fully_obs is the reference that fixes the task directions
CHECKPOINT_FOLDERS = {'fully_obs': 'results', 'allocentric': 'results_allocentric', 'pf': 'results_pf'}
SPEC = {}  # environment spec, filled in main, read by forked workers


# ---------------------------------------------------------------- environment specs

def pp_success(last):
    return bool(last[4]['__common__']['agent0']['success'])


def football_success(last):  # kept possession until the step limit
    return bool(last[3]['__all__']) and not bool(last[2]['__all__'])


def drone_success(last):  # survived to the step limit without breach / out of bounds
    c0 = last[4]['__common__']['agent0']
    return bool(last[3]['__all__']) and not (c0['breached'] or c0['oob'])


def fire_success(last):  # fire extinguished or contained before the step limit (green cells always remain)
    return bool(last[2]['__all__']) and not bool(last[3]['__all__'])


ENVS = {
    'predator_prey': dict(
        agents=['agent0', 'agent1', 'agent2'], p=2, adversary='prey', own_slice=slice(0, 6), n_actions=5,
        success=pp_success, extras=[],
        checkpoint='logs/marl/predator_prey_fully_obs/checkpoint1000'),
    'football': dict(
        agents=['agent0', 'agent1', 'agent2', 'agent3', 'agent4'], p=2, adversary='opponent', own_slice=slice(0, 6),
        n_actions=11, success=football_success, extras=[('ball', 'ball_pos')],
        checkpoint='logs/marl/football_fully_obs/checkpoint3000'),
    'drones': dict(
        agents=['agent0', 'agent1', 'agent2'], p=3, adversary='intruder', own_slice=slice(0, 21), n_actions=11,
        success=drone_success, extras=[],
        checkpoint='logs/marl/drones_fully_obs/checkpoint1500'),
    # fire: the state is an image, so z / z_hat are the policy's own conv latent of the true / received image
    'fire': dict(
        agents=['agent0', 'agent1', 'agent2', 'agent3'], p=2, adversary='fire', n_actions=10,
        success=fire_success, extras=[], latent=True, obs_range=30, blue=4, n_fire_pcs=20,
        checkpoint='logs/marl/fire_fully_obs/checkpoint5000'),
    # market: one scalar (stockpile) per teammate, sorted because the belief's teammate slots are permutation
    # invariant; y is each company's (order, sell) fraction, decoded from the 13 x 51 flattened action
    'market_game': dict(
        agents=[f'agent{i}' for i in range(7)], p=1, adversary='release', own_slice=slice(0, 5), n_actions=663,
        success=pp_success, extras=[], action_fractions=dict(n_sell=51, order_step=0.05, sell_step=0.02),
        sort_team=True, scale=100.0, units='material units',
        centroid_axes=("true teammates' average stockpile", "true spread (std) of teammates' stockpiles"),
        checkpoint='logs/marl/market_game_7a_fully_obs/checkpoint829'),
}


def title():
    if SPEC.get('latent'):
        return f"policy conv latent of the {SPEC['adversary']} image ({SPEC['latent_dim']}-d)"
    n_mates, dim = len(SPEC['agents']) - 1, SPEC['p']
    extras = [name for name, _ in SPEC['extras']]
    plus = ''.join(f' + {e}' for e in extras)
    return f"teammates + {SPEC['adversary']}{plus} ({(n_mates + 1 + len(extras)) * dim}-d)"


# ---------------------------------------------------------------- data

def policy_targets(record):
    """y: every agent's action from the policy on its true observation, one-hot with the last action dropped
    (the columns of a full one-hot sum to 1, which makes their covariance singular)."""
    acts = record[4]['__common__']['policy_actions_true_obs']
    if SPEC.get('action_fractions'):
        af = SPEC['action_fractions']
        return np.concatenate([np.array(divmod(int(np.asarray(acts[a]).item()), af['n_sell']), dtype=float)
                               * [af['order_step'], af['sell_step']] for a in SPEC['agents']])
    n = SPEC['n_actions']
    return np.concatenate([np.eye(n)[int(np.asarray(acts[a]).item())][:n - 1] for a in SPEC['agents']])


def load_episode(path, max_steps=None, with_y=False):
    """Variables per step. Record k holds the state after step k but the actions chosen before it (from
    the state in record k - 1), so the action responding to the state in record k is read from record
    k + 1; with_y drops the final step, which has no next action."""
    agents = SPEC['agents']
    d = np.load(path, allow_pickle=True)
    obs_map = d['obs_map'].item()
    steps_all = sorted(int(k) for k in d.files if k.isdigit())
    steps = steps_all[:max_steps]
    if with_y and steps[-1] == steps_all[-1]:
        steps = steps[:-1]
    obs = {a: [] for a in agents}
    true = {a: [] for a in agents}
    pred = {a: [] for a in agents}
    y = []
    for k in steps:
        o, _, _, _, info = d[str(k)]
        common = info['__common__']
        nn = {a: np.asarray(common['obs_no_noise'][a], dtype=np.float64) for a in agents}
        sp = common.get('sampled_predictions', {})
        y.append(policy_targets(d[str(k + 1)]) if with_y else np.zeros(0))
        for a in agents:
            obs[a].append(np.asarray(o[a], dtype=np.float64))
            true[a].append(nn[a])
            pred[a].append(np.asarray(sp.get(a, o[a][obs_map['team']]), dtype=np.float64))
    y = np.array(y)
    out = {}
    scale = SPEC.get('scale', 1.0)
    for a in agents:
        ob, tr = np.array(obs[a]), np.array(true[a])
        team, team_hat = tr[:, obs_map['team']], np.array(pred[a]).reshape(len(steps), -1)
        if SPEC.get('sort_team'):
            team, team_hat = np.sort(team, axis=1), np.sort(team_hat, axis=1)
        out[a] = dict(team=scale * team, team_hat=scale * team_hat,
                      adv=scale * tr[:, obs_map['target_pos']], adv_hat=scale * ob[:, obs_map['target_pos']],
                      own=tr[:, SPEC['own_slice']], y=y)
        for name, key in SPEC['extras']:
            out[a][name] = tr[:, obs_map[key]]
            out[a][f'{name}_hat'] = ob[:, obs_map[key]]
    return out


def episode_files(results_dir, limit=None):
    return sorted(glob.glob(os.path.join(results_dir, '*.npz')),
                  key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))[:limit]


def load_method(results_dir, workers, max_steps, limit=None, with_y=False):
    """Variable sets of one method, all agents pooled and ordered by episode. with_y (the fully-observed
    reference only) also reads the policy actions that fix the task directions."""
    files = episode_files(results_dir, limit)
    with Pool(workers) as p:
        eps = p.map(partial(load_episode, max_steps=max_steps, with_y=with_y), files)
    agents = SPEC['agents']
    per_agent = []
    for a in agents:
        da = {k: np.concatenate([e[a][k] for e in eps]) for k in eps[0][a]}
        da['ep'] = np.concatenate([np.full(len(e[a]['team']), i) for i, e in enumerate(eps)])
        per_agent.append(da)
    d = {k: np.concatenate([da[k] for da in per_agent]) for k in per_agent[0]}
    order = np.argsort(d['ep'], kind='stable')
    d = {k: v[order] for k, v in d.items()}
    extras = [name for name, _ in SPEC['extras']]
    out = dict(z=np.column_stack([d['team'], d['adv']] + [d[e] for e in extras]),
               zh=np.column_stack([d['team_hat'], d['adv_hat']] + [d[f'{e}_hat'] for e in extras]),
               own=d['own'], y=d['y'], adv=d['adv'], ep=d['ep'])
    out.update({e: d[e] for e in extras})
    return out


def episode_rows(ep):
    bounds = np.flatnonzero(np.diff(ep)) + 1
    return np.split(np.arange(len(ep)), bounds)


# ---------------------------------------------------------------- fire: policy latent of the image

FIRE_EPISODES = {}  # method -> per-episode (agree, compared, success, length), for action_metrics


def fire_image(dec, noise=False):
    """The agent's 61x61 observation image rebuilt from decomposed_obs, exactly as DroneFireEnv._combine_obs
    draws it: its clean fire window plus a cross at each teammate within range. noise=False puts the crosses
    at the TRUE teammate positions (the env adds +-7 cells of noise, even to the logged obs_no_noise image)."""
    R, blue = SPEC['obs_range'], SPEC['blue']
    w = 2 * R + 1
    img = np.array(dec['fire'], dtype=np.uint8)
    offsets = dec['noise'] if noise else [(0, 0)] * len(dec['team'])
    for tp, nz in zip(dec['team'], offsets):
        r, c = tp[0] + nz[0], tp[1] + nz[1]
        if abs(r) <= R and abs(c) <= R:
            for rr, cc in [(r + R, c + R), (r + R - 1, c + R), (r + R + 1, c + R), (r + R, c + R - 1), (r + R, c + R + 1)]:
                if 0 <= rr < w and 0 <= cc < w:
                    img[rr, cc] = blue
    return img


def load_fire_episode(path, max_steps=None, reference=False):
    """Per agent-step: the true and the received image, true and estimated teammate positions, own position
    and the action taken from this state (read from the next record; -1 at the final step)."""
    agents = SPEC['agents']
    d = np.load(path, allow_pickle=True)
    steps_all = sorted(int(k) for k in d.files if k.isdigit())
    steps = steps_all[:max_steps]
    rows = {k: [] for k in ['true_img', 'recv_img', 'team', 'team_hat', 'own', 'taken'] + (['fire'] if reference else [])}
    for k in steps:
        o, _, _, _, info = d[str(k)]
        common = info['__common__']
        nxt = d[str(k + 1)][4]['__common__']['policy_actions'] if k + 1 in steps_all else {}
        for a in agents:
            dec = common['decomposed_obs'][a]
            team = np.ravel(np.asarray(dec['team'], dtype=np.float64))
            rows['true_img'].append(fire_image(dec))
            rows['recv_img'].append(np.asarray(o[a]).reshape(o[a].shape[:2]).astype(np.uint8))
            rows['team'].append(team)
            # fully observed: the noisy teammate positions it observes; beliefs: their own estimates
            hat = common['sampled_predictions'][a] if 'sampled_predictions' in common else (
                team + np.ravel(np.asarray(dec['noise'], dtype=np.float64)))
            rows['team_hat'].append(np.ravel(np.asarray(hat, dtype=np.float64)))
            rows['own'].append(np.asarray(dec['self_pos'], dtype=np.float64))
            rows['taken'].append(int(np.asarray(nxt[a]).item()) if a in nxt else -1)
            if reference:
                rows['fire'].append(np.asarray(dec['fire'], dtype=np.uint8).ravel())
    out = {k: np.array(v) for k, v in rows.items()}
    out['success'] = SPEC['success'](d[str(steps_all[-1])])
    out['length'] = len(steps_all)
    return out


def load_fire_policy(checkpoint_dir, device):
    """The fire policy's torch model (RLlib VisionNetwork, one policy shared by all agents)."""
    import sys
    import torch
    from ray.rllib.policy.policy import Policy
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    pid = sorted(os.listdir(os.path.join(checkpoint_dir, 'policies')))[0]
    model = Policy.from_checkpoint(os.path.abspath(checkpoint_dir), policy_ids=[pid])[pid].model
    model.eval().to(device)
    modules = list(model._convs.children())
    cut = next(i for i, mod in enumerate(modules) if isinstance(mod, torch.nn.Flatten))
    # conv stack up to its flatten: the same latent as the fire UMAP analysis (umap_fire.py)
    return model, torch.nn.Sequential(*modules[:cut + 1])


def encode(images, model, encoder, device, batch=4096):
    """Conv latent and the policy's greedy action (explore=False) for each image."""
    import torch
    lat, act = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch, None].astype(np.float32)).to(device)
            lat.append(encoder(x).cpu().numpy())
            act.append(model._logits(model._convs(x)).reshape(len(x), -1).argmax(1).cpu().numpy())
    return np.concatenate(lat).astype(np.float64), np.concatenate(act)


def load_fire_raw(results_dir, workers, max_steps, limit, reference):
    files = episode_files(results_dir, limit)
    with Pool(workers) as p:
        return p.map(partial(load_fire_episode, max_steps=max_steps, reference=reference), files)


def fire_sets(eps, results_dir, reference, model, encoder, device):
    """Variable sets for fire. z / z_hat = the policy's conv latent of the true / received image; y (reference
    only) = the policy's own action on the true image, one-hot. Latent units that never activate on the
    reference data are dropped (SPEC['latent_keep'])."""
    n_agents = len(SPEC['agents'])
    d = {k: np.concatenate([e[k] for e in eps]) for k in eps[0] if k not in ('success', 'length')}
    d['ep'] = np.concatenate([np.full(len(e['team']), i) for i, e in enumerate(eps)])
    z, true_act = encode(d.pop('true_img'), model, encoder, device)
    zh, _ = encode(d.pop('recv_img'), model, encoder, device)
    if reference:
        SPEC['latent_keep'] = z.std(0) > 1e-6
        SPEC['latent_dim'] = int(SPEC['latent_keep'].sum())
    keep = SPEC['latent_keep']
    out = dict(z=z[:, keep], zh=zh[:, keep], own=d['own'], team=d['team'], team_hat=d['team_hat'], ep=d['ep'])
    if reference:
        n = SPEC['n_actions']
        # the joint action of every agent at that step, as in the other environments: rows come in groups of
        # one step (agent0..agent3), so each group's one-hots (last action dropped) are concatenated and
        # given to all of its rows
        joint = np.eye(n)[true_act][:, :n - 1].reshape(-1, n_agents * (n - 1))
        out['y'] = np.repeat(joint, n_agents, axis=0)
        fire = d['fire'].astype(np.float64)
        fire -= fire.mean(0)
        _, _, Vt = np.linalg.svd(fire[::max(1, len(fire) // 20000)], full_matrices=False)
        out['fire_pc'] = fire @ Vt[:SPEC['n_fire_pcs']].T
    # action agreement: the action taken from this state vs the policy's action on the true image
    valid = d['taken'] >= 0
    agree = (d['taken'] == true_act) & valid
    FIRE_EPISODES[results_dir] = np.array([[agree[d['ep'] == i].sum(), valid[d['ep'] == i].sum(), e['success'],
                                            e['length']] for i, e in enumerate(eps)], dtype=float)
    return out


# ---------------------------------------------------------------- linear algebra

def inv_sqrt(S, eps=1e-6):
    lam, Q = np.linalg.eigh(S + eps * np.eye(len(S)))
    return Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T


def residualize(x, own):
    """Remove the linear dependence on the agent's own state."""
    X = np.column_stack([own - own.mean(0), np.ones(len(own))])
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta


def standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def cca(a, b, eps=1e-6):
    """Canonical correlations and the weights of each side, via SVD."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    n = len(a)
    Wa, Wb = inv_sqrt(a.T @ a / n, eps), inv_sqrt(b.T @ b / n, eps)
    U, s, Vt = np.linalg.svd(Wa @ (a.T @ b / n) @ Wb, full_matrices=False)
    return s, Wa @ U, Wb @ Vt.T


def unit(W):
    return W / np.linalg.norm(W, axis=0)


def r2(t, X):
    """R^2 of t linearly predicted by X (with intercept), used for the composition only."""
    Xc = np.column_stack([X - X.mean(0), np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(Xc, t, rcond=None)
    return float(1 - np.var(t - Xc @ beta) / np.var(t))


def group_bases(d_raw, n_mates, p, blocks=('adversary',)):
    """Orthonormal bases (raw coordinates) of the centroid / spread subspaces and each later block."""
    mate_dims = n_mates * p
    cen = np.zeros((d_raw, p))
    for j in range(n_mates):
        cen[j * p:(j + 1) * p] = np.eye(p)
    cen /= np.sqrt(n_mates)
    proj = np.eye(d_raw)[:, :mate_dims] - cen @ (cen.T @ np.eye(d_raw)[:, :mate_dims])
    U, s, _ = np.linalg.svd(proj, full_matrices=False)
    out = {'centroid': cen, 'difference': U[:, s > 1e-8]}
    for b, name in enumerate(blocks):
        out[name] = np.eye(d_raw)[:, mate_dims + b * p:mate_dims + (b + 1) * p]
    return out


def teammates(s):
    """True and estimated teammate positions, (n, n_mates * p) each."""
    if SPEC.get('latent'):
        return s['team'], s['team_hat']
    k = (len(SPEC['agents']) - 1) * SPEC['p']
    return s['z'][:, :k], s['zh'][:, :k]


def centroid(team):
    n_mates, p = len(SPEC['agents']) - 1, SPEC['p']
    return team.reshape(len(team), n_mates, p).mean(1)


def physical_bases(z):
    blocks = ['adversary'] + [name for name, _ in SPEC['extras']]
    return group_bases(z.shape[1], len(SPEC['agents']) - 1, SPEC['p'], blocks)


def composition(F, s):
    """For unit raw functionals F (columns): R^2 from each physical group, and the geometric share
    of each group's subspace (the row labels in the figures). For a latent state (fire) there are no
    physical subspaces: each direction is labelled by the R^2 with which the true teammate positions and
    the fire window (its top principal components) linearly explain it."""
    if SPEC.get('latent'):
        zc = s['z'] - s['z'].mean(0)
        feats = {'teammates': s['team'], 'fire': s['fire_pc']}
        return {'explained': {g: [r2(zc @ f, X) for f in F.T] for g, X in feats.items()}}
    n_mates, p = len(SPEC['agents']) - 1, SPEC['p']
    z = s['z']
    zc = z - z.mean(0)
    mates = z[:, :n_mates * p].reshape(len(z), n_mates, p)
    cen = mates.mean(1)
    feats = {'adversary': s['adv'], 'centroid': cen, 'difference': (mates - cen[:, None]).reshape(len(z), -1)}
    feats.update({name: s[name] for name, _ in SPEC['extras']})
    bases = physical_bases(z)
    out = {'r2': {g: [] for g in feats}, 'share': {g: [] for g in bases}}
    for f in F.T:
        t = zc @ f
        for g, X in feats.items():
            out['r2'][g].append(r2(t, X))
        for g, Bg in bases.items():
            out['share'][g].append(float(np.sum((Bg.T @ f) ** 2)))
    return out


# ---------------------------------------------------------------- analysis

def task_cca(s):
    """CCA of the whitened true state with the fully-observed policy's actions, own state regressed out of both.
    Returns the squared canonical correlations and the unit task functionals in raw coordinates."""
    zc = s['z'] - s['z'].mean(0)
    W = inv_sqrt(np.cov(zc, rowvar=False))
    zt_res = residualize(zc @ W, s['own'])
    st, Tw, _ = cca(zt_res, standardize(residualize(s['y'], s['own'])))
    return st ** 2, unit(W @ unit(Tw))


def axes(sets, min_eig, r_ref=None):
    """Task functionals from the true model (fixed for every method)."""
    eig, F_all = task_cca(sets['fully_obs'])
    if r_ref is None:
        r_ref = int((eig > min_eig).sum())
    return r_ref, F_all[:, :r_ref], eig, F_all


def r2_along(err, truth):
    """Multivariate R^2 = 1 - sum of squared errors / total variance (columns already projected)."""
    return float(1 - np.sum(err ** 2) / np.sum((truth - truth.mean(0)) ** 2))


def evaluate(sets, min_eig, r_ref=None):
    """Raw R^2 of each belief (its own estimate, exactly what the policy receives) on the fixed axes."""
    r_ref, F_task, eig, F_all = axes(sets, min_eig, r_ref)
    groups = None if SPEC.get('latent') else physical_bases(sets['fully_obs']['z'])
    out = {'r_ref': r_ref, 'task_eig': eig[:r_ref], 'task_eig_all': eig}
    for m in METHODS:
        z, zh = sets[m]['z'], sets[m]['zh']
        err = zh - z
        task = np.array([r2_along(err @ F_task[:, [k]], z @ F_task[:, [k]]) for k in range(r_ref)])
        out[m] = {'raw': dict(
            task=task,
            task_all=np.array([r2_along(err @ F_all[:, [k]], z @ F_all[:, [k]]) for k in range(F_all.shape[1])]),
            coverage=float(task.mean()),  # mean over the fixed controller task directions
            physical=({g: r2_along(err @ B, z @ B) for g, B in groups.items()} if groups is not None else
                      # latent state: only the teammate centroid, from the positions themselves
                      {'centroid': r2_along(centroid(teammates(sets[m])[1]) - centroid(teammates(sets[m])[0]),
                                            centroid(teammates(sets[m])[0]))}),
        )}
    return out


def flatten(x, prefix=''):
    """Flatten numeric leaves to {path: array}."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f'{prefix}{k}/'))
    elif x is not None:
        out[prefix.rstrip('/')] = np.asarray(x, dtype=float)
    return out


def to_json(x):
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, list, tuple)):
        return np.asarray(x, dtype=float).round(5).tolist()
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    return x


# ---------------------------------------------------------------- figure data

def projections(sets, F, n_keep, seed):
    """Truth and the raw belief estimate projected on the fixed task directions."""
    rng = np.random.default_rng(seed)
    out = {'functionals': F}
    for m in METHODS:
        z, zh = sets[m]['z'], sets[m]['zh']
        idx = rng.choice(len(z), size=min(n_keep, len(z)), replace=False)
        out[f'{m}/actual'] = z[idx] @ F
        out[f'{m}/belief'] = zh[idx] @ F
    return out


def centroid_samples(sets, n_keep, seed):
    """True and raw belief teammate centroid (x, y; relative to the agent) for the centroid map."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in METHODS:
        team, team_hat = teammates(sets[m])
        if SPEC['p'] == 1:
            # scalar centroid: map over (centroid, true teammate spread); both use the true spread, so the
            # error and the R^2 denominator are the centroid's alone
            spread = team.std(1)
            full, full_hat = np.column_stack([centroid(team), spread]), np.column_stack([centroid(team_hat), spread])
            total_var = np.var(full[:, 0])
        else:
            full, full_hat = centroid(team)[:, :2], centroid(team_hat)[:, :2]
            total_var = np.sum(np.var(full, 0))
        idx = rng.choice(len(full), size=min(n_keep, len(full)), replace=False)
        out[f'{m}/true'] = full[idx]
        out[f'{m}/raw'] = full_hat[idx]
        # total variance of the full centroid over all samples: the R^2 denominator
        out[f'{m}/total_var'] = np.array(total_var)
    return out


def episode_r2(sets, F):
    """Per-episode raw R^2 along each task functional: 1 - mean episode squared error / variance of the
    truth over ALL of that method's data (so episodes are comparable and average to the overall R^2)."""
    out = {}
    for m in METHODS:
        z, zh, ep = sets[m]['z'], sets[m]['zh'], sets[m]['ep']
        t, e = z @ F, (zh - z) @ F
        var = t.var(0)
        rows = episode_rows(ep)
        out[m] = np.stack([1 - (e[r] ** 2).mean(0) / var for r in rows])  # (n_episodes, n_directions)
        out[f'{m}/n_steps'] = np.array([len(r) for r in rows])
    return out


def save_figure_data(sets, est, min_eig, out_dir, seed):
    _, F_task, _, _ = axes(sets, min_eig)
    np.savez(os.path.join(out_dir, 'episode_r2.npz'), **episode_r2(sets, F_task))
    proj = projections(sets, F_task, 80000, seed)
    proj['task_eig'] = est['task_eig']
    for kind, groups in composition(F_task, sets['fully_obs']).items():
        if kind in ('share', 'explained'):  # row labels: subspace shares, or R^2 explained (latent state)
            for g, v in groups.items():
                proj[f'{kind}/{g}'] = np.array(v)
    np.savez(os.path.join(out_dir, 'projections.npz'), **proj)
    np.savez(os.path.join(out_dir, 'centroid.npz'), **centroid_samples(sets, 200000, seed))


# ---------------------------------------------------------------- actions and success

def episode_actions(path):
    """Per episode: steps where the action taken equals the policy's action on the true observation,
    agent-steps compared, success flag and length (full episode)."""
    d = np.load(path, allow_pickle=True)
    steps = sorted(int(k) for k in d.files if k.isdigit())
    agree = total = 0
    for k in steps:
        c = d[str(k)][4]['__common__']
        taken, true = c.get('policy_actions', {}), c.get('policy_actions_true_obs', {})
        for a, act in taken.items():
            if a in true:
                total += 1
                agree += int(np.asarray(act).item() == np.asarray(true[a]).item())
    return agree, total, SPEC['success'](d[str(steps[-1])]), len(steps)


def action_metrics(dirs, n_boot, seed, workers):
    """Action agreement and success rate per method, with an episode bootstrap."""
    rng = np.random.default_rng(seed)
    est, boot = {}, {}
    for m, dpath in dirs.items():
        if SPEC.get('latent'):  # computed while loading (the policy's action on the rebuilt true image)
            r = FIRE_EPISODES[dpath]
        else:
            files = sorted(glob.glob(os.path.join(dpath, '*.npz')))
            with Pool(workers) as pool:
                r = np.array(pool.map(episode_actions, files), dtype=float)
        if r[:, 1].sum() == 0:
            print(f'{m}: no action logging in {dpath}', flush=True)
            continue
        def stats(i):
            return np.array([r[i, 0].sum() / r[i, 1].sum(), r[i, 2].mean(), r[i, 3].mean()])
        est[m] = stats(np.arange(len(r)))
        boot[m] = np.stack([stats(rng.integers(len(r), size=len(r))) for _ in range(n_boot)])
    names = ['action_agreement', 'success_rate', 'mean_episode_length']
    return ({m: dict(zip(names, v.tolist())) for m, v in est.items()},
            {f'actions/{m}/{n}': boot[m][:, i] for m in boot for i, n in enumerate(names)})


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=list(ENVS), required=True)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--min-eig', type=float, default=0.1)
    p.add_argument('--n-boot', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit-episodes', type=int, default=None, help='for testing only')
    p.add_argument('--out-dir', default=None, help='default results/<env>/cca_final')
    p.add_argument('--data-dir', default=None,
                   help='checkpoint folder holding results/, results_allocentric/, results_pf/ '
                        '(default: the final checkpoint in ENVS)')
    p.add_argument('--device', default='cuda', help='fire only: device for the policy encoder')
    p.add_argument('--figdata-only', action='store_true',
                   help='only rewrite episode_r2.npz, projections.npz and centroid.npz '
                        '(no bootstrap, keeps the existing summary)')
    args = p.parse_args()

    SPEC.update(ENVS[args.env])
    out_dir = args.out_dir or os.path.join('results', args.env, 'cca_final')
    os.makedirs(out_dir, exist_ok=True)
    data_dir = args.data_dir or SPEC['checkpoint']
    dirs = {m: os.path.join(data_dir, CHECKPOINT_FOLDERS[m]) for m in METHODS}

    sets = {}
    if SPEC.get('latent'):
        # read every method's episodes before loading the policy, so the worker pool never forks a CUDA process
        raw = {m: load_fire_raw(dirs[m], args.workers, args.max_steps, args.limit_episodes, m == 'fully_obs')
               for m in METHODS}
        model, encoder = load_fire_policy(data_dir, args.device)
    for m in METHODS:  # fully_obs first: it fixes the reference (and, for fire, the latent units kept)
        if SPEC.get('latent'):
            sets[m] = fire_sets(raw.pop(m), dirs[m], m == 'fully_obs', model, encoder, args.device)
        else:
            sets[m] = load_method(dirs[m], args.workers, args.max_steps, args.limit_episodes,
                                  with_y=m == 'fully_obs')
        print(f'loaded {m}: {len(sets[m]["ep"])} agent-steps', flush=True)

    est = evaluate(sets, args.min_eig)
    save_figure_data(sets, est, args.min_eig, out_dir, args.seed)
    if args.figdata_only:
        print('FIGURE DATA DONE', flush=True)
        return

    rng = np.random.default_rng(args.seed)
    rows = {m: episode_rows(sets[m]['ep']) for m in METHODS}
    boots = []
    for b in range(args.n_boot):
        bs = {}
        for m in METHODS:
            idx = np.concatenate([rows[m][i] for i in rng.integers(len(rows[m]), size=len(rows[m]))])
            bs[m] = {k: v[idx] for k, v in sets[m].items()}
        boots.append(flatten(evaluate(bs, args.min_eig, r_ref=est['r_ref'])))
        if b % 20 == 0:
            print(f'bootstrap {b}/{args.n_boot}', flush=True)
    ci = {k: np.percentile(np.stack([bb[k] for bb in boots]), [2.5, 97.5], axis=0).tolist() for k in boots[0]}
    np.savez(os.path.join(out_dir, 'bootstrap.npz'), **{k: np.stack([bb[k] for bb in boots]) for k in boots[0]})

    _, F_task, _, F_all = axes(sets, args.min_eig)
    actions, actions_boot = action_metrics(dirs, args.n_boot, args.seed, args.workers)
    print('actions:', json.dumps(actions), flush=True)
    np.savez(os.path.join(out_dir, 'bootstrap_actions.npz'), **actions_boot)

    summary = {
        'settings': dict(vars(args), out_dir=out_dir, data_dirs=dirs),
        'actions': actions,
        'meta': dict(env=args.env, adversary=SPEC['adversary'], title=title(),
                     n_teammates=len(SPEC['agents']) - 1, latent=bool(SPEC.get('latent')),
                     units=SPEC.get('units', 'grid cells' if SPEC.get('latent') else 'env units'),
                     centroid_axes=SPEC.get('centroid_axes')),
        'task_composition': to_json(composition(F_task, sets['fully_obs'])),
        'task_composition_all': to_json(composition(F_all, sets['fully_obs'])),
        'estimate': to_json(est),
        'ci95': ci,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'saved {out_dir}/summary.json', flush=True)
    print('ANALYSIS DONE', flush=True)


if __name__ == '__main__':
    main()
