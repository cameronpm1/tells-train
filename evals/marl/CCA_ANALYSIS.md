# CCA analysis: allocentric belief vs particle filter

This analysis asks how well each teammate-belief model reproduces the parts of the world state that the task depends on. It covers the allocentric belief network and the egocentric particle filter (PF), in predator-prey, football and drones.

The yardstick comes from the decentralized fully observed (FO) RL policy:
1. On the FO rollouts, CCA between the true state and the FO policy's actions gives the **task directions**. These are the fixed linear functionals of the state that the policy's actions depend on most.
2. Each belief model is scored by the **raw R²** of its own estimate along those same directions.

Scores use no fitting or calibration, so they measure exactly what the policy receives.

## Data

The data for each environment comes from one final checkpoint, evaluated three ways with `run.py` (`marl_eval`, `marl_eval_belief`, `pf_eval`, with `--collect_results`):

| Environment | Checkpoint | Teammates | Adversary | Extra entity | Dimensions |
|---|---|---|---|---|---|
| predator_prey | `logs/marl/predator_prey_fully_obs/checkpoint1000` | 2 | prey | – | 2 |
| football | `logs/marl/football_fully_obs/checkpoint3000` | 4 | opponent | ball | 2 |
| drones | `logs/marl/drones_fully_obs/checkpoint1500` | 2 | intruder | – | 3 (x, y, altitude) |

Each checkpoint folder holds the three evaluation sets, with 1000 episodes per method:

| Method | Folder |
|---|---|
| FO | `results/` |
| allocentric | `results_allocentric/` |
| PF | `results_pf/` |

The analysis uses the first 200 steps of each episode, with every agent pooled as an agent-step.

For each agent-step:

- **z (truth):** the teammates, the adversary and any extra entity, relative to the agent, taken from `obs_no_noise`.
- **ẑ (estimate):** the same quantities as the agent's belief sees them:
  - teammates come from the belief's `sampled_predictions`, or the observation if those are absent;
  - adversary and extra entity come from the observation the policy receives.

  For PF, the adversary and the ball are the filter's estimates. For FO, ẑ is the noisy observation itself:
  - predator-prey: σ = 0.7;
  - drones: σ = 0.3;
  - football: no noise.
- **own:** the agent's own state, which is regressed out of the task CCA:
  - predator-prey: velocity, position, goal;
  - football: velocity, position, anchor;
  - drones: velocity, position, box positions.
- **y (policy actions, FO data only):** every agent's discrete action from the FO RL checkpoint, taken on its true (noise-free) observation:
  - logged as `policy_actions_true_obs` by `eval.py`;
  - encoded one-hot, with the last action dropped because the full one-hot columns sum to 1;
  - all agents stacked: 5 actions per agent in predator-prey, 11 in football and drones.

  Record k of an episode file holds the state *after* step k but the actions chosen *before* it, which responded to record k−1's state. So the state in record k is paired with the actions in record k+1, and the final step of each FO episode is dropped. This pairing gives stronger correlations than same-record pairing.

### Fire: the policy's latent instead of positions

In fire, the observation is an image, so the state is taken in the policy's own representation of it.

| | Data | Notes |
|---|---|---|
| Checkpoint | `logs/marl/fire_fully_obs/checkpoint5000` | 4 agents, 10 actions, 1000 episodes per method |
| Image | the agent's 61×61 local fire window, with a cross at each teammate within 30 cells | as drawn by `DroneFireEnv._combine_obs` |
| **z** | the policy's conv latent of the **true** image | conv stack up to its flatten layer, the same layer as the fire UMAP analysis (`umap_fire.py`); latent units never active on FO data are dropped |
| **ẑ** | the same latent of the image the policy **received** | FO: teammate crosses with ±7 cells of noise. Allocentric: crosses at the belief's estimates. PF: the filter's whole image, fire window included. |
| **own** | the agent's grid position | |
| **y** | the FO policy's greedy action on the true image, one-hot, all agents stacked | computed directly, so there is no step offset |

The true image is rebuilt from `decomposed_obs`: the clean fire window, plus crosses at the true teammate positions. That rebuild is needed because the logged `obs_no_noise` image already carries the env's ±7-cell teammate noise. With the logged noise added back, the rebuild reproduces the env's image exactly. Running the policy on the stored images reproduces the logged actions in 99.9–100% of agent-steps.

- **Row labels:** latent directions have no physical subspaces. Each is labelled by the R² with which the true teammate positions, and the fire window (top 20 principal components on FO data), linearly explain it.
- **Centroid:** from the true teammate positions against each belief's estimate (`sampled_predictions`), in grid cells.
- **Success:** the fire was extinguished or contained before the 125-step limit.
- **Action agreement:** the action taken against the policy's action on the rebuilt true image.

## Method

### 1. Task directions (fixed, from FO data)

On the FO data, z is whitened and own state is regressed out of both z and y. The analysis then runs a CCA between the residualised actions y and the whitened state.

The FO data supplies both sides: the states the FO rollouts visited, and the FO policy's actions in them. The allocentric and PF data are never used to find the directions, only scored along them.

- A direction counts as a task direction if its squared canonical correlation σ² is above 0.1 (`--min-eig`).
- Each task direction is mapped back to raw state coordinates as a unit functional `f_k`.

| Environment | Task directions | σ² |
|---|---|---|
| predator-prey | 2 | 0.21, 0.18 |
| football | 9 | 0.60, 0.56, 0.33, 0.24, 0.24, 0.19, 0.16, 0.13, 0.11 |
| drones | 3 | 0.43, 0.36, 0.31 |

The **same** `f_k` are applied to every method's data. No method gets its own axes.

Each direction is labelled by its composition: the share of `f_k` that lies in each physical subspace.
- **centroid:** mean of the teammates.
- **spread:** teammates minus their centroid.
- **adversary.**
- **ball:** football only.

### 2. Raw R² along a fixed direction

For one functional `f` and one method:

```
R² = 1 − mean((ẑ − z)·f)² / var(z·f)
```

This is the belief's own estimate, not a regression on it. It is 1 for a perfect estimate and 0 for an estimate no better than the mean, and it is unbounded below. A noisy or biased estimate on a low-variance direction can therefore go far below 0.

**Coverage** is the mean R² over the task directions. The same functionals give the physical R²s (centroid, spread, adversary, ball) through orthonormal bases of each subspace. Both have 95% intervals from an episode-level bootstrap (100 resamples), stored in `summary.json` and `bootstrap.npz`.

### 3. Per-episode R² (ridgeline)

For each episode and each task direction:

```
R²_episode = 1 − mean over the episode of ((ẑ − z)·f)² / var(z·f) over all of that method's episodes
```

The denominator is shared, so episodes are comparable and their step-weighted mean is the pooled R².

### 4. Centroid R² by position (centroid map)

This uses the teammate centroid in x, y relative to the agent. For drones it is horizontal only.

- Samples are binned on the **true** centroid position (a grid of regular hexagons, 70 across, over one square range covering the 1st–99th percentile of both x and y, so the axes share a scale).
- In each bin:

  ```
  local R² = 1 − mean ‖ĉ − c‖² in the hexagon / total centroid variance
  ```

- The denominator is shared, so the count-weighted mean of the hexagons is the overall centroid R².
- Hexagons with fewer than 20 samples are blank.

## Figures

Both figures are in `results/<env>/cca_final/`.

### `task_ridgeline.png`: per-episode R² along each task direction

- **Rows:** one per task direction, strongest first. Each row label shows:
  - σ²;
  - composition (shares ≥ 5%);
  - median per-episode R² for allocentric and PF.
- **Curves:** allocentric (blue) and PF (orange) share the row's baseline. Fills are see-through, so a curve behind the other stays visible. Each curve is a KDE of the 1000 per-episode R² values, scaled to its own peak.
- **x-axis:** runs from max(−1, 1st percentile rounded down) to 1. Values below the axis start are piled at the left edge. For example, in football the axis starts at −0.4.
- **How to read it:** a curve further right means better recovery, and a wider curve means less consistency across episodes.
- **FO is left out.** Its R² reflects observation noise rather than a belief: football FO is exactly 1, while predator-prey and drone FO are below 1 because of their observation noise.

### `centroid_map.png`: local centroid R² over the true centroid position

- Two panels, allocentric and PF (egocentric), on a shared colour scale. The scale's lower end is the 2nd percentile of the hexagons, floored at −1.
- Each panel title gives the overall centroid R².
- The map shows *where* in teammate configuration space each belief loses the centroid, for example when teammates are far away or outside the observation window.

### Other outputs

`plot_cca_analysis.py` also writes outputs that are not part of the converged set:

| Output | Contents |
|---|---|
| `fixed_axis.png` | violins of coverage, task and physical R² |
| `fixed_axis_spectrum.png` | the spectrum |
| `pred_vs_true.png` | predicted vs true values |
| `actions.png` | action agreement and success |
| `task_error_ridgeline.png` | ridgeline of the error distribution |

## Results (raw)

Coverage and centroid R² are pooled over all agent-steps. Medians are over episodes.

| | Predator-prey | Football | Drones |
|---|---|---|---|
| Coverage: FO / allocentric / PF | 0.97 / 0.93 / 0.45 | 1.00 / 0.77 / 0.63 | 0.76 / 0.94 / 0.41 |
| Pooled R² by task direction: allocentric | 0.93, 0.93 | 0.67, 0.80, 0.73, 0.75, 0.88, 0.83, 0.92, 0.69, 0.65 | 0.95, 0.92, 0.94 |
| Pooled R² by task direction: PF | 0.43, 0.46 | 0.76, 0.19, 0.72, 0.59, 0.76, 0.68, 0.78, 0.54, 0.66 | 0.64, 0.36, 0.22 |
| Median per-episode R² by task direction: allocentric | 0.95, 0.95 | 0.80, 0.88, 0.84, 0.82, 0.92, 0.90, 0.94, 0.79, 0.78 | 0.96, 0.98, 0.98 |
| Median per-episode R² by task direction: PF | 0.79, 0.77 | 0.93, 0.87, 0.95, 0.74, 0.88, 0.85, 0.94, 0.84, 0.89 | 0.69, 0.50, 0.32 |
| Centroid R²: allocentric / PF | 0.52 / −0.18 | 0.91 / 0.77 | 0.90 / 0.71 |
| Action agreement: FO / allocentric / PF | 69% / 64% / 37% | 100% / 63% / 18% | 95% / 85% / 40% |
| Success: FO / allocentric / PF | 100% / 92% / 13% | 67% / 16% / 0% | 99% / 75% / 10% |

**Fire** (policy latent, 7 task directions, σ² 0.77, 0.75, 0.75, 0.57, 0.45, 0.33, 0.29):

| | FO | Allocentric | PF |
|---|---|---|---|
| Coverage (95% CI) | 0.77 [0.74, 0.78] | 0.88 [0.87, 0.90] | 0.47 [0.44, 0.53] |
| Pooled R² by task direction | 0.86, 0.86, 0.87, 0.68, 0.74, 0.73, 0.65 | 0.88, 0.88, 0.87, 0.87, 0.89, 0.97, 0.82 | 0.08, 0.51, 0.59, 0.60, 0.48, 0.67, 0.37 |
| Median per-episode R² | – | 0.91, 0.89, 0.88, 0.89, 0.90, 0.99, 0.85 | 0.56, 0.63, 0.57, 0.62, 0.61, 0.95, 0.51 |
| Centroid R² (grid cells) | 0.81 | −1.53 | −3.85 |
| Action agreement | 79% | 76% | 52% |
| Success | 99% | 71% | 49% |

- **What the directions reflect.** The true teammate positions explain at most 5% of any fire task direction. The fire window's top 20 principal components explain 11–46%, and the rest is non-linear structure in the latent. So the fire directions are about how the policy reads the fire, not where its teammates are.
- **Why allocentric beats PF.** The allocentric image keeps the true fire window and only replaces the teammate crosses. PF rebuilds the whole image, fire included, from its filter.
- **Allocentric above FO.** Allocentric coverage is above FO's. Two things contribute:
  - FO's image has ±7-cell noise on its teammate crosses;
  - each method's R² is normalised by the variance of its own episodes, and FO's episodes are much shorter (18 steps against 53).

Notes:
- **Action agreement** is the fraction of agent-steps where the action taken equals the action the same policy would take on the true observation. The two are logged as `policy_actions` and `policy_actions_true_obs` by `eval.py` / `eval_pf.py`.
- **Success** is defined per environment:
  - predator-prey: capture;
  - football: possession kept to the step limit;
  - drones: reached the step limit with no breach or out-of-bounds.

### Caveats

- **Pooled vs median.** Pooled R² is dominated by a few very bad episodes, so it can sit well below the per-episode median. PF predator-prey shows this: coverage is 0.45, while the medians are about 0.78. The ridgeline shows both: the body of each curve and the tail piled at the left edge.
- **Where PF beats allocentric (football only).**
  - Task direction 1 (53% centroid, 28% ball, 18% opponent): PF is better on the pooled R² as well, 0.76 vs 0.67, and the bootstrap interval of the difference, [0.04, 0.13], excludes 0.
  - Directions 3, 8 and 9: PF has the higher per-episode median, because allocentric has more mediocre episodes. Pooled, these directions are tied or favour allocentric.

  Allocentric is better on every direction in predator-prey and drones, with intervals excluding 0.
- **Drone altitude.** The drones hold an almost constant altitude (true std ≈ 0.04). With the policy-action targets, none of the three drone task directions is altitude-heavy: all are mostly intruder plus centroid. All three dimensions are still kept in z.
- **Football ball.** The allocentric ball R² is 1.0, because the predicted-ball field written into the observation matches the *current* true ball position. PF's ball R² is 0.45. This affects the football task directions with a ball component, mainly task directions 1 (28% ball) and 3 (21%).

## Reproducing

`results/run_cca_final.sh` runs the full analysis, including the bootstrap, then all figures for the three environments in parallel. Each run logs to `results/<env>/cca_final/run.log`. The per-environment command is:

```bash
PY=/home/cameron/miniforge3/envs/tells/bin/python
$PY evals/marl/cca_analysis.py --env football      # default --data-dir: the checkpoint in the table above
$PY plotting/plot_cca_analysis.py --in-dir results/football/cca_final
```

`--out-dir` defaults to `results/<env>/cca_final`.

There are two shorter paths:
- **Figures only, from saved data:** the `plot_cca_analysis.py` line above.
- **Figure data only:** `results/run_raw_figures.sh`. It adds `--figdata-only`, which rewrites `episode_r2.npz`, `projections.npz` and `centroid.npz` without the bootstrap and keeps `summary.json`.

### Outputs in `results/<env>/cca_final/`

| File | Contents |
|---|---|
| `summary.json` | task σ², raw R² per task direction, coverage, physical R², composition, 95% CIs, action metrics, data paths |
| `bootstrap.npz`, `bootstrap_actions.npz` | bootstrap samples |
| `episode_r2.npz` | per-episode R², (episodes × task directions) per method; source of the ridgeline |
| `projections.npz` | task functionals, σ², composition shares, sampled truth/estimate projections |
| `centroid.npz` | sampled true and estimated centroids, total centroid variance; source of the centroid map |

## Code

| File | Role |
|---|---|
| [cca_analysis.py](cca_analysis.py) | the whole analysis, in sections. |
| [plotting/plot_cca_analysis.py](../../plotting/plot_cca_analysis.py) | all figures from the saved outputs. The two figures used are drawn by `plot_r2_ridgeline` and `plot_centroid_map`. |

The sections of `cca_analysis.py`:

| Section | Contents |
|---|---|
| environment specs | `ENVS`: agents, dimensions, adversary, own-state slice, number of actions, extras, success test, final checkpoint |
| data | `load_episode` and `load_method`: z, ẑ and own state, all agents pooled; y from `policy_targets` for the FO data |
| linear algebra | `cca`, `inv_sqrt`, `residualize`, `standardize`; the physical subspaces (`group_bases`) and the composition of each direction (`composition`) |
| analysis | `task_cca` (task directions from FO data), `evaluate` (raw R², coverage, physical R²), bootstrap |
| figure data | `episode_r2`, `projections`, `centroid_samples` |
| actions and success | `action_metrics` |
