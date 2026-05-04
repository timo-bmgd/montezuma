import os

from tqdm import trange
import imageio
import gymnasium as gym
import numpy as np
import torch

from agilerl.algorithms import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs
from agilerl.rollouts.on_policy import collect_rollouts

import gymnasium as gym
import ale_py

# Initial hyperparameters
INIT_HP = {
    "POP_SIZE": 4,  # Population size
    "BATCH_SIZE": 128,  # Batch size
    "LR": 0.001,  # Learning rate
    "LEARN_STEP": 1024,  # Learning frequency
    "GAMMA": 0.99,  # Discount factor
    "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
    "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
    "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
    "ENT_COEF": 0.01,  # Entropy coefficient
    "VF_COEF": 0.5,  # Value function coefficient
    "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
    "TARGET_KL": None,  # Target KL divergence threshold
    "UPDATE_EPOCHS": 4,  # Number of policy update epochs
    "TARGET_SCORE": 200.0,  # Target score that will beat the environment
    "MAX_STEPS": 150000,  # Maximum number of steps an agent takes in an environment
    "EVO_STEPS": 10000,  # Evolution frequency
    "EVAL_STEPS": None,  # Number of evaluation steps per episode
    "EVAL_LOOP": 3,  # Number of evaluation episodes
    "TOURN_SIZE": 2,  # Tournament size
    "ELITISM": True,  # Elitism in tournament selection
}

# Mutation parameters
MUT_P = {
    # Mutation probabilities
    "NO_MUT": 0.4,  # No mutation
    "ARCH_MUT": 0.2,  # Architecture mutation
    "NEW_LAYER": 0.2,  # New layer mutation
    "PARAMS_MUT": 0.2,  # Network parameters mutation
    "ACT_MUT": 0.2,  # Activation layer mutation
    "RL_HP_MUT": 0.2,  # Learning HP mutation
    "MUT_SD": 0.1,  # Mutation strength
    "RAND_SEED": 42,  # Random seed
}

# RL hyperparameters configuration for mutation during training
hp_config = HyperparameterConfig(
    lr = RLParameter(min=1e-4, max=1e-2),
    batch_size = RLParameter(min=8, max=1024),
)


gym.register_envs(ale_py)

# Initialise the environment
num_envs = 8
env = make_vect_envs("ALE/MontezumaRevenge-v5", num_envs=num_envs)  # Create environment

observation_space = env.single_observation_space
action_space = env.single_action_space

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
