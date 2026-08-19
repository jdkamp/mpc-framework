import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "python" / "envs"))
sys.path.append(str(REPO_ROOT / "build"))
sys.path.append(str(REPO_ROOT / "scripts"))

from lane_keeping_env import LaneKeepingEnv
from train_safe_learning import build_filter

def probe_trajectory(safety_filter, steps=40):
    probe = LaneKeepingEnv(k_true=0.0, sigma_scale=0.0)   # noise-free: isolate the filter
    probe.reset(seed=0)
    probe.x = np.array([0.0, 0.0, 0.0, 10.0])
    ys = []
    for _ in range(steps):
        x = probe.x
        u = safety_filter.filter(np.asarray(x), np.array([0.3, 0.0]))  # full lock
        probe.x = probe._dynamics(x, u)
        ys.append(probe.x[1])                                          # store py
    return ys

gen_dir = REPO_ROOT / "output" / "safe_learning"
params  = sorted(gen_dir.glob("gp_gen*.json"))            # gen0 (cold), gen1, ...
cmap    = plt.cm.viridis

for i, p in enumerate(params):
    ys = probe_trajectory(build_filter(p))
    plt.plot(ys, color=cmap(i / max(len(params) - 1, 1)), label=f"gen {i}")

plt.axhline(2.0, ls="--", color="red", label="true bound")
plt.xlabel("step"); plt.ylabel("py [m]")
plt.title("Safe set grows: the filter lets the agent nearer the bound each generation", fontsize=10)
plt.legend(); plt.tight_layout()
plt.savefig(gen_dir / "safe_learning_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()
