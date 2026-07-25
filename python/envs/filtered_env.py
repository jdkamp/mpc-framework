import gymnasium as gym
import numpy as np

class SafetyFilterWrapper(gym.Wrapper):
    """Passes every agent action through a safety filter before the plant sees it"""
    
    def __init__(self, env, safety_filter, bicycle=None):
        super().__init__(env)
        self.filter = safety_filter
        self.bicycle = bicycle  # mpc_py.Bicycle, for the nominal yaw rate
        self.data = []          # collected (v, delta, residual) data

    def step(self, action):
        x = self.env.unwrapped.x    # Full [px, py, psi, v] before the step
        u_safe = self.filter.filter(np.asarray(x), np.asarray(action, dtype=np.float64))
        obs, r, term, trunc, info = self.env.step(u_safe)

        if self.bicycle is not None:
            x_next = self.env.unwrapped.x
            dt = self.env.unwrapped.dt
            observed_yaw_rate = (x_next[2] - x[2]) / dt
            nominal_yaw_rate = self.bicycle.dynamics(np.asarray(x), u_safe)[2]
            residual = observed_yaw_rate - nominal_yaw_rate
            self.data.append([x[3], u_safe[0], residual])       # v, delta, residual

        return obs, r, term, trunc, info