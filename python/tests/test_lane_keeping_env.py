import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "python" / "envs"))

from lane_keeping_env import LaneKeepingEnv
from gp_sigma import GPSigma


def test_parity_with_cpp():
    ref = np.loadtxt(REPO_ROOT / "data" / "parity_reference.csv", delimiter=",", skiprows=1)
    env = LaneKeepingEnv(k_true=0.0, sigma_scale=0.0)   # pure nominal bicycle
    env.reset(seed=0)
    env.x = np.array([0.0, 0.5, 0.2, 10.0])            # same start as the C++ main
    u = np.array([0.05, 0.2])
    for t in range(100):
        assert np.allclose(env.x, ref[t], atol=1e-12), f"diverged at step {t}"
        env.x = env._dynamics(env.x, u)

def test_check_env():
    from gymnasium.utils.env_checker import check_env
    check_env(LaneKeepingEnv())

def test_gp_sigma_matches_cpp():
    gp = GPSigma()
    for v, d, var_ref in [(10, 0.3, 0.001710), (3, -0.2, 0.000004), (8, 0.0, 0.000033)]:
        var = gp.sigma(v, d)**2
        assert abs(var - var_ref) < 1e-5, f"mismatch at ({v}, {d}): {var:.6f} vs {var_ref}"

def test_pd_holds_lane():
    env = LaneKeepingEnv()
    obs, _ = env.reset(seed=0)
    k_p, k_d = 0.1, 1.0

    max_py = 0.0
    for t in range(500):
        delta = -k_p * obs[0] - k_d * obs[1]
        obs, r, term, trunc, info = env.step(np.array([delta, 0.0]))
        max_py = max(max_py, abs(obs[0]))
        if term or trunc:
            break

    assert not info ["violation"], f"PD controller violated at step {t}"
    assert t == 499, f"episode ended early at step {t}"
    assert max_py < 2.0, f"max |py| = {max_py:.2f} touched the bound"


def test_random_actions_crash():
    env = LaneKeepingEnv()
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)

    for t in range(500):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break

    assert t<499, "random policy survived the full episode - task may be too easy"
    assert info["violation"], "episode ended but not by violation"