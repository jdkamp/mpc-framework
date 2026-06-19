import pandas as pd
import matplotlib.pyplot as plt

DT = 0.1
PY_BOUND = 6.0

runs = {
    "Deterministic MPC": ("output/cmp_detmpc.csv", "tab:red"),
    "GP-MPC"           : ("output/cmp_gpmpc.csv",  "tab:orange"),
    "GP-SMPC"          : ("output/cmp_smpc.csv",   "tab:green"),
}

data = {name: pd.read_csv(path) for name, (path, _) in runs.items()}

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# 1) py vs time -- respecting the constraint?
for name, (_, color) in runs.items():
    df = data[name]
    axes[0].plot(df["t"] * DT, df["py"], color=color, label=name)
axes[0].axhline(PY_BOUND, color="black", linestyle="--", label="py constraint")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("py (lateral position)")
axes[0].set_title("Lateral position vs constraint")
axes[0].legend()
axes[0].grid(True)

# 2) path px-py -- spatial view of the maneuver
for name, (_, color) in runs.items():
    df = data[name]
    axes[1].plot(df["px"], df["py"], color=color, label=name)
axes[1].axhline(PY_BOUND, color="black", linestyle="--")
axes[1].set_xlabel("px")
axes[1].set_ylabel("py")
axes[1].set_title("Path")
axes[1].legend()
axes[1].grid(True)

# 3) steering -- shows the GP controllers act differently from deterministic
for name, (_, color) in runs.items():
    df = data[name]
    axes[2].plot(df["t"] * DT, df["delta"], color=color, label=name)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Steering (delta)")
axes[2].set_title("Steering input")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("output/comparison.png", dpi=120)
print("Saved output/comparison.png")
plt.show()
