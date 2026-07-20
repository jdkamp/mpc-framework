import gymnasium as gym
import numpy as np

class SafetyFilterWrapper(gym.Wrapper):
    """Passes every agent action through a safety filter before the plant sees it"""
    
    def __init__(self, env, safety_filter):
        super().__init__(env)
        self.filter = safety_filter

    def step(self, action):
        x = self.env.unwrapped.x    # Full [px, py, psi, v] before the step
        u_safe = self.filter.filter(np.asarray(x), np.asarray(action, dtype=np.float64))
        return self.env.step(u_safe)