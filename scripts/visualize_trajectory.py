import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/trajectory.csv")

fix, axes = plt.subplots(5, 1, figsize=(10, 10))

# Path
axes[0].plot(df["px"], df["py"], label="Actual")
axes[0].plot(df["px_ref"], df["py_ref"], color="red", linestyle="--", label="Reference")
axes[0].set_xlabel("px")
axes[0].set_ylabel("py")
axes[0].legend()
axes[0].grid(True)

# Heading
axes[1].plot(df["t"], df["psi"], label="Actual")
axes[1].plot(df["t"], df["psi_ref"], color="red", linestyle="--", label="Reference")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Heading (psi)")
axes[1].legend()
axes[1].grid(True)

# Steering input
axes[2].plot(df["t"], df["delta"])
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Steering (delta)")
axes[2].grid(True)

# Velocity
axes[3].plot(df["t"], df["v"], label="Actual")
axes[3].plot(df["t"], df["v_ref"], color="red", linestyle="--", label="Reference")
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("v")
axes[3].legend()
axes[3].grid(True)

# Acceleration
axes[4].plot(df["t"], df["a"])
axes[4].set_xlabel("Time (s)")
axes[4].set_ylabel("a")
axes[4].grid(True)

for ax in axes:
    ax.set_xlim(left=0)



plt.tight_layout()
plt.show()

