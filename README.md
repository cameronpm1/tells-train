# Multi-Agent Training Environment (Communications-Denied Scenarios)

## Overview
This repository provides a training environment for collaborative multi-agent behavior in communications-denied environments.

## Repository Layout
This repository is the code half of the `tells_results` folder. Clone it into `tells_results/tells-train`; the
results, models, videos and run scripts sit next to it:

```
tells_results/
├── tells-train/      this repository: run.py, confs/, envs/, learn/, evals/, plotting/, util/ (+ data/, logs/, git-ignored)
├── scripts/          bash pipelines: train_<env>.sh, collect_eval_data.sh, analyze_env.sh, noise_sweep_{run,plot}.sh
├── models/           marl/<run>/checkpoint<N>/ policies, belief/_<run>_NN_<i>/epoch<N>/ belief models
├── results/          <env>/results_{fully_obs,allocentric,pf}/, cca/, noise_sweep/, controller_results/
└── videos/           supplementary grid GIFs / MP4s
```

The analysis and plotting code finds `results/` and `models/` through `util/util.py` (`RESULTS_ROOT`, `MODELS_ROOT`),
which default to the folder above this repository; set `TELLS_RESULTS=/path/to/tells_results` to point them elsewhere.
Paths inside `confs/` (`logdir`, `data_dir`, `log_dir`) are relative to this repository, so run everything from here.

## Current Support
- MARL (Multi-Agent Reinforcement Learning) codebase
- Uses RLlib with Soft Actor-Critic (SAC) or PPO for training
- Supports both MPE2 predator prey environment and pybullet-drones training environment


The default sync installs the predator-prey/MARL stack. The pybullet drone
environment is required for the caravan env. We recommend using a conda environment as it is necessary for google-football. Google-football has not been updated in several years, and requires an outdated version of setup-tools, as a result installation can be tricky depending on your machine. Set up instructions acter cloning git-repository below:

```bash
git submodule update --init --recursive

conda env create -f environment.yml

conda activate tells

python -m pip install "pip<23" "setuptools==65.5.0" "wheel<0.40"

python -m pip install -r requirements.txt

pip install -e external/pybullet-drones
```

Check if google-football is if not download it from git manually:

```bash
cd external

git clone git@github.com:google-research/football.git

cd football

pip install -r requirements.txt

python -m pip install .
```

If google-football hits an error in install that says "likely not an error with pip," it could me an issue with you environment pointing to newer versions of boost and setup tools outside of your conda environment. In order to fix this, run the following command, you can copy and paste the entire prompt into your linux terminal. The rerun "python -m pip install .".

```bash
cp gfootball/build_game_engine.sh gfootball/build_game_engine.sh.bak

python - <<'PY'
from pathlib import Path

p = Path("gfootball/build_game_engine.sh")
s = p.read_text()

old = "cmake ."
new = (
    "cmake . "
    "-DPython_EXECUTABLE=$CONDA_PREFIX/bin/python "
    "-DPython_ROOT_DIR=$CONDA_PREFIX "
    "-DPython_FIND_STRATEGY=LOCATION "
    "-DPython_FIND_VIRTUALENV=FIRST "
    "-DBOOST_ROOT=$CONDA_PREFIX "
    "-DBoost_NO_SYSTEM_PATHS=ON "
    "-DCMAKE_PREFIX_PATH=$CONDA_PREFIX "
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
)

if old not in s:
    raise SystemExit("Could not find plain 'cmake .' in build_game_engine.sh")

p.write_text(s.replace(old, new))
print("patched")
PY

rm -rf build
rm -rf third_party/gfootball_engine/CMakeFiles
rm -f third_party/gfootball_engine/CMakeCache.txt
```

---

## Usage

### Training a Fully-Observable Model
```bash
python run.py --command marl_train --config confs/predator_prey/3a_game.yaml
```
or 
```bash
python run.py --command marl_train --config confs/drones/3a_game.yaml
```
- Starts training using the specified configuration file, look in `confs/` to see all environment config files
- Logs are stored in: `logs/marl/` (will autogenerate the directory if not already created)

---

### Evaluation
```bash
python run.py --command marl_eval --config confs/predator_prey/3a_game.yaml --model_dir logs/marl/test12/checkpoint3500 --runs 1
```
- Evaluates a trained checkpoint for a specified number of runs  
- Generates video outputs  
- Saves videos in a `video/` folder inside the checkpoint directory  

---

### Tensorboard 
```bash
tensorboard --logdir logs/marl/test12/
```
- Shows tensorboard log files for specified directory 

---

### Collect Data to Train a Belief Model

- Data for the belief model is stored inside a data/ directory
-Data collection can be run by using the collect_data command:

```bash
python run.py --command marl_collect_data --config confs/predator_prey/3a_game.yaml --model_dir logs/marl/ppotest1/checkpoint10000 --runs 800 --n_workers 8
```

- runs is the number of epsisodes to collect data from

- n_workers is the number of parallel environments to create when collecting data (important unless you want to wait a while)

---

### Training a Belief Model

- Once data is collected, a model can be trained by running the following command: 

```bash
python run.py --command belief_train --config confs/belief/predator_prey.yaml
```

---

### Evaluating a Belief Model with an RL Policy

- Model evaluation can be done by running:

```bash
python run.py --command marl_eval_belief --belief_dir logs/belief/predator_prey_belief/lightning_logs/checkpoints/{epoch}_{val_loss} --belief_config confs/belief/pre.yaml --config confs/predator_prey/3a_game.yaml --model_dir logs/marl/ppo_bc_ex/checkpoint4500/
```

- This will create a directory called test_outputs inside the model_dir and save several test videos to it. In order to edit how the videos are created/saved look at .test_save() inside the belief model class

---

### Retrain RL Policy with Belief Model

- Model evaluation can be done by running:

```bash
python run.py --command marl_train_belief --belief_dir logs/belief/predator_prey_belief/lightning_logs/checkpoints/{epoch}_{val_loss} --belief_config confs/belief/predator_prey.yaml --config confs/predator_prey/3a_game_allocentric.yaml --model_dir logs/marl/predator_prey/checkpoint5000/
```

- retrains the fully-observable RL policy in the communication denied environment using the belief model. Adds variational free energy loss term. Pay attention to config files! Retraining requires a config with `allocentric` in its name. You must use the same config file when evaluating the trained model with `--command marl_train_belief`

## Environments
- `PredatorPreyScenario` is based on openai's mpe environemnt:

  - Several layers of classes:
  ```
  PredatorPreyEnv (Gym Environment)
      └── ScenarioEnv
              └── PredatorPreyScenario
  ```

- `CaravanAviary` uses pybullet-drones to create a caravan protection env

- `CirclePass5v1Env` built off google research football

  - Custom rondo environment defined in `build_scenario`

- `DroneFireEnv` custom gym environment for drone fire-fighting scenario

  - `DroneFireSim` handles environemnt dynamics

  ```
  DroneFireEnv (Gym Environment)
      └── DroneFireSim
  ```

- `MarketGameEnv` custom gym environment for market coordination game

  - Companies buy material from a scripted supplier and sell product to meet demand

- `MarketGameLargeEnv` scaled version of `MarketGameEnv` with 60 companies

  - Teammates are observed through quantiles of their stockpiles

- `PredatorPreySwarmEnv` scaled version of `PredatorPreyEnv` with 40 predators

  - Teammates are observed through the swarm's covariance ellipse

  - Several layers of classes:
  ```
  PredatorPreySwarmEnv (Gym Environment)
      └── SwarmScenarioEnv
              └── PredatorPreySwarmScenario
  ```

### RLlib Integration
- All environments are wrapped with:
  ```
  RLlibWrapper
  ```
- Ensures compatibility with RLlib MultiAgentEnv interface  
- Used for both training and evaluation pipelines  

### Important Environment Structures

- Unique environment parameters should always exist inside the gym envirnment, the RLlibWrapper should act as a generic wrapper for all environments

- Each gym environment must have a parameter obs_map which is a dictionary of labels and slices. Labels must match terminology in current environments if integrating a new one -- important for ensuring RLlibWrapper functions correctly

- All gym environmnents must provide action/observation space per agent, not for entire environment. Full environment action/observation space is compiled in RLlibWrapper. This is uncommon for gym environments so worthy of noting for future developing.

## Testing Baselines

### Evaluate Egocentric Estimator

```bash
python run.py --command ego_eval --model_dir logs/marl/predator_prey_pf/checkpoint2000 --config confs/predator_prey/3a_game_pf.yaml
```

---

## Notes
- Designed for experimentation in decentralized, communication-restricted coordination tasks  
- Modular structure allows easy integration for additional scenarios and agent behaviors  
