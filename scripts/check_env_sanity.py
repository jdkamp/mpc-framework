import sys; sys.path.append("python/envs")
import numpy as np
import matplotlib.pyplot as plt
from lane_keeping_env import LaneKeepingEnv

env = LaneKeepingEnv()
obs, _ = env.reset(seed=0)
traj = []
for t in range(500):
    # PD controller to lane center: py = obs[0], py_dot = v*sin(psi) ~ psi=obs[1]
    k_p, k_d = 0.1, 1.0
    delta = -k_p * obs[0] - k_d * obs[1]
    
    obs, r, term, trunc, info = env.step(np.array([delta, 0.0]))
    traj.append(obs[0])
    if term or trunc: break

print("steps survived:", len(traj), "| max |py|: %.2f" % max(abs(p) for p in traj))

plt.plot(traj)
plt.axhline(2, ls="--", c="k"); plt.axhline(-2, ls="--", c="k")
plt.xlabel("step"); plt.ylabel("py")
plt.savefig("output/env_sanity.png")
print("saved output/env_sanity.png")
plt.show()
