import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

# Add the environment package to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'FR3_env'))

# Import the environment
from panda_mujoco_gym.envs import FrankaPickAndPlaceEnv

# Create the environment
env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="human")
env = DummyVecEnv([lambda: env])

# Create the model
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./franka_tensorboard/"
)

# Train the model
total_timesteps = 1_000_000  # Adjust this based on your needs
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("franka_ppo_model")

# Test the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()