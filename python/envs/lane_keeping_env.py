import gymnasium as gym
import numpy as np
from gp_sigma import GPSigma

class LaneKeepingEnv(gym.Env):
    """Kinematic bicycle, lane keeping under disturbance

    World:        straight lane, symmetric bound |py| <= py_max = 2.
    dt:           0.1
    Speed regime: v_ref = 10 — the GP is uncertain there (training data centers on v≈5),
                  so sigma and the filter's backoff are actually visible.
    Disturbance:  GP-based sigma(v, delta) on psi (from data/gp_params.json), plus the
                  true residual -k*v^2*delta.
    Observation:  [py, psi, v] — px is irrelevant for lane keeping.
    Action:       [delta, a], bounds copied from MPCLimits (u_min/u_max).
    Reward:       r = +1  - 1.0*py^2  - 0.05*(v - v_ref)^2  - 0.01*(delta^2 + a^2)
                  - +1 alive bonus: with only negative terms, terminating early would be
                    optimal ("crashing is cheap") — surviving inside the lane must pay.
                  - speed term: kills the standing-still exploit (v=0 makes lane keeping
                    trivial) and pins the env in the uncertain GP regime.
                  - violation: episode terminates at |py| > py_max; losing the future
                    alive-stream is the penalty (no extra reward spike). info["violation"]
                    is logged every step for the with/without-filter comparison.
                  - shared by both experiments: the filtered agent's reward still varies
                    (py, v, effort) even if it never violates.
    """

    def __init__(self, dt=0.1, wheelbase=2.7, py_max=2, v_ref=10.0, k_true=0.004, sigma_scale=1.0):
        self.dt, self.L, self.py_max, self.v_ref, self.k_true = dt, wheelbase, py_max, v_ref, k_true
        self.gp = GPSigma()
        self.x = None
        self.steps = 0
        self.w_y, self.w_v, self.w_u = 1.0, 0.05, 0.01
        self.sigma_scale = sigma_scale

        self.action_space = gym.spaces.Box(
            low = np.array([-0.5, -5.0], dtype=np.float32),
            high = np.array([0.5, 5.0], dtype=np.float32))
        self.observation_space = gym.spaces.Box(
            low = -np.inf, high = np.inf, shape=(3,), dtype=np.float32)

    def _dynamics(self, x, u):
        px, py, psi, v = x
        delta, a = u
        dx = np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            v * np.tan(delta) / self.L + self._residual(v, delta),
            a
        ])
        x_next = x + self.dt * dx
        x_next[2] += self.np_random.normal(0.0, self.sigma_scale * self.gp.sigma(v, delta))
        return x_next

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        py0 = self.np_random.uniform(-1.0, 1.0)
        self.x = np.array([0.0, py0, 0.0, self.v_ref])
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.x = self._dynamics(self.x, action)
        self.steps += 1
        py, v = self.x[1], self.x[3]
        delta, a = action
        reward = 1.0 - self.w_y * py**2 - self.w_v * (v - self.v_ref)**2 - self.w_u * (delta**2 + a**2)
        violated = bool(abs(py) > self.py_max)
        return self._get_obs(), float(reward), bool(violated), self.steps >= 500, {"violation": bool(violated)}
    
    def _get_obs(self):
        return np.array([self.x[1], self.x[2], self.x[3]], dtype=np.float32)
    
    def _residual(self, v, delta):
        return -self.k_true * v**2 * delta