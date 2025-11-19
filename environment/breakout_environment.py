import gymnasium as gym
import cv2
import ale_py
import numpy as np

class BreakoutEnv:
    def __init__(self, render_mode=None):
        gym.register_envs(ale_py)
        self.env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        self.observation_space = (84, 84, 1)
        self.action_space = self.env.action_space

    def preprocess(self, obs):
        obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
        obs_normalized = obs_resized / 255.0
        return np.expand_dims(obs_normalized, axis=-1).astype(np.float32)

    def reset(self):
        obs, info = self.env.reset()
        return self.preprocess(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return self.preprocess(obs), reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()