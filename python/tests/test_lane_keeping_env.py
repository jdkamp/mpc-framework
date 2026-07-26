import sys
from pathlib import Path
import numpy as np
import json
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "python" / "envs"))
sys.path.append(str(REPO_ROOT / "build"))

mpc_missing = not any((REPO_ROOT / "build").glob("mpc_py*.so"))

from lane_keeping_env import LaneKeepingEnv
from gp_sigma import GPSigma

@pytest.mark.skipif(mpc_missing, reason="mpc_py not built (run cmake --build build)")
def test_parity_with_cpp():
    import mpc_py
    env = LaneKeepingEnv(k_true=0.0, sigma_scale=0.0)   # pure nominal bicycle
    bicycle = mpc_py.BicycleModel(2.7)
    x = np.array([0.0, 0.5, 0.2, 10.0])
    u = np.array([0.05, 0.2])
    for t in range(100):
        x_cpp = x + env.dt * bicycle.dynamics(x, u)
        x_py = env._dynamics(x, u)
        assert np.allclose(x_py, x_cpp, atol=1e-12), f"diverged at step {t}"
        x = x_py

def test_check_env():
    from gymnasium.utils.env_checker import check_env
    check_env(LaneKeepingEnv())

def test_gp_sigma_matches_cpp():
    params = json.load(open(REPO_ROOT / "data" / "gp_params.json"))
    gp = GPSigma()
    for chk in params["reference_checks"]:
        var = gp.sigma(chk["v"], chk["delta"])**2
        assert abs(var - chk["var"]) < 1e-5, f"mismatch at ({chk['v']}, {chk['delta']})"
        

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