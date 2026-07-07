import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DT, PY_BOUND = 0.1, 9.0
R, V_REF = 8.0, 10.0 # must match main.cpp

runs = {
    "Deterministic MPC": ("output/ens_detmpc.csv", "tab:red"),
    "GP-MPC":            ("output/ens_gpmpc.csv",  "tab:orange"),
    "GP-SMPC":           ("output/ens_smpc.csv",   "tab:green"),
}

def reference_py(T):
    t = np.arange(T)
    s = V_REF * DT * t
    return t, R * (1.0 - np.cos(s / R))

# Top row: full view (shows the runaways). Bottom row: zoomed to the band
# around the bound, so the crossings (= violations) are clearly visible.
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

for col, (name, (path, color)) in enumerate(runs.items()):
    df = pd.read_csv(path)
    for row in range(2):
        ax = axes[row, col]
        for _, g in df.groupby("rollout"):
            ax.plot(g["t"] * DT, g["py"], color=color, alpha=0.15)   # one faint line = one rollout
        t_ref, py_ref = reference_py(int(df["t"].max()) + 1)
        ax.plot(t_ref * DT, py_ref, color="tab:blue", lw=1.5, label="reference")
        ax.axhline(PY_BOUND, color="black", linestyle="--", label="py constraint")
        ax.grid(True)
    axes[0, col].set_title(name)
    axes[0, col].set_ylim(0, 16.5)   # full view (captures the runaways)
    axes[1, col].set_ylim(8, 10)   # zoom: the band around the 9.0 bound
    axes[1, col].set_xlabel("Time (s)")

axes[0, 0].set_ylabel("py — full view")
axes[1, 0].set_ylabel("py — zoom near bound")
axes[0, 2].legend(loc="upper left")

plt.suptitle("Monte-Carlo trajectory ensemble vs constraint")
plt.tight_layout()
plt.savefig("output/ensemble.png", dpi=120)
print("Saved output/ensemble.png")
plt.show()
