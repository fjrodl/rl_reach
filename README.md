# RL Reach
RL Reach is a platform for running reproducible reinforcement learning experiments. Training environments are provided to solve the reaching task with the WidowX MK-II robotic arm.
The Gym environments and training scripts are adapted from [Replab](https://github.com/bhyang/replab) and [Stable Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), respectively.

![Alt text](/docs/widowx_env.gif?raw=true "The Widowx Gym environment in Pybullet")


4 training environments are available:
- widowx_reacher-v1: fixed goal
- widowx_reacher-v2: fixed goal, goal-oriented environment (compatible with HER)
- widowx_reacher-v3: random goal
- widowx_reacher-v4: random goal, goal-oriented environment (compatible with HER)

## Installation


1. Clone the repository

```bash
git clone https://github.com/PierreExeter/rl_reach.git && cd rl_reach/
```

2. Install and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate rl_reach
```
Note, this environment assumes that you have CUDA 11.1 installed. If you are using another version of CUDA, you will have to install Pytorch manually as indicated [here](https://pytorch.org/get-started/locally/).

3. Install the custom Gym environments

```bash
cd gym_envs/
pip install -e .
```

Alternatively, use the Docker container (see section below).

## Test the installation

Manual tests

```bash
python tests/manual/1_test_widowx_env.py
python tests/manual/2_test_train.py
python tests/manual/3_test_enjoy.py
python tests/manual/4_test_pytorch.py
```

Automated tests

```bash
pytest tests/auto/all_tests.py -v
```


## Train RL agents

RL experiments can be launched with the script `run_experiments.py`.

Usage:
- `--exp-id`: Unique experiment ID (int)
- `--algo`: RL algorithm (str: a2c, ddpg, her, ppo, sac, td3)
- `--env`: Training environment ID (str: widowx_reacher-v1, widowx_reacher-v2, widowx_reacher-v3, widowx_reacher-v4)
- `--n-timesteps`: Number of training steps (int)
- `--n-seeds`: Number of runs with different initialisation seeds (int)

Example:
```bash
python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-seeds 5
```
Run all experiments:
```bash
./run_all_exp.sh
```

## Evaluate policy and save results

Trained models can be evaluated and the results can be saved with the script `evaluate_policy.py`.

Usage:
- `--exp-id`: Unique experiment ID (int)
- `--n-eval-steps`: Number of evaluation timesteps (int)
- `--log-info`: Log information at each evaluation steps and save (0 or 1)
- `--plot-dim`: Plot end effector and goal position in real time (0: Don't plot (default), 2: 2D, 3: 3D)
- `--render`: Render environment during evaluation (0 or 1)

Example:
```bash
python evaluate_policy.py --exp-id 99 --n-eval-steps 1000 --log-info 0 --plot-dim 0 --render 0
```

Environment evaluation plot:

![Alt text](/docs/plot_episode_eval_log.png)

Experiment learning curves:

![Alt text](/docs/reward_vs_timesteps_smoothed.png)

## Benchmark

The evaluation metrics, environment's variables, hyperparameters used during the training and parameters for evaluating the environments are logged for each experiments in the file `benchmark/benchmark_results.csv`. Evaluation metrics of selected experiments ID can be plotted with the `script scripts/plot_benchmark.py`.

Usage:
- `--exp-list`: List of experiments to consider for plotting (list of int)
- `--col`: Name of the hyperparameter for the X axis, see column names in `benchmark/benchmark_results.csv` (str)

Example:
```bash
python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps
```

## Optimise hyperparameters

Hyperparameters can be tuned with the script `train.py -optimize`.


Usage:
- `--algo`: RL algorithm (str: a2c, ddpg, her, ppo, sac, td3)
- `--env`: Training environment ID (str: widowx_reacher-v1, widowx_reacher-v2, widowx_reacher-v3, widowx_reacher-v4)
- `--n-timesteps`: Number of training steps (int)
- `--n-trials`: Number of optimisation trials (int)
- `--n-jobs`: Number of parallel jobs (int)
- `--sampler`: Sampler for optimisation search (str: random, tpe, skopt)
- `--pruner`: Pruner to kill unpromising trials early (str: halving, median, none)
- `--n-startup-trials`: Number of trials before using optuna sampler (int)
- `--n-evaluations`: Number of episode to evaluate a trial (int) 
- `--log-folder`: Log folder for the results (str) 

Example:
```bash
python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti
```

Optimise all experiments:
```bash
./opti_all.sh
```

### Docker images (work in progress)

Pull image

or 

build image yourself

```bash
./docker/build_docker_cpu.sh
./docker/build_docker_gpu.sh
```

Run commands inside the docker container

```bash
./docker/run_docker_cpu.sh python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 30000 --n-seeds 2
./docker/run_docker_gpu.sh python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 30000 --n-seeds 2
```



## Tested on

- Ubuntu 18.04
- Python 3.7.9
- Conda 4.9.2
- CUDA 11.1