from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "output"

runs = {
    "SAC (unfiltered)":     (OUT / "baseline_violations.csv", "tab:red"),
    "SAC + safety filter":  (OUT / "filtered_violations.csv", "tab:green")
}

plt.figure(figsize=(8, 5))
for label, (path, color) in runs.items():
    if not path.exists():
        print(f"({label}: {path.name} not found - skipped)")
        continue
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    plt.plot(data[:, 0], data[:, 1], color=color, label=label)

plt.xlabel("Training step")
plt.ylabel("Cumulative constraint violations")
plt.title("Cost of learning: violations during RL training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT / "training_violations.png", dpi=120)
print("Saved", OUT / "training_violations.png")
plt.show()