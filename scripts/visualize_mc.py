import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/mc_violations.csv")

colors = {"Deterministic MPC": "tab:red", "GP-MPC": "tab:orange", "SMPC": "tab:green"}
bars = plt.bar(df["controller"], df["rate_pct"],
               color=[colors[c] for c in df["controller"]])

plt.axhline(5.0, color="black", linestyle="--", label="1 - p = 5% budget")
plt.ylabel("Constraint violation rate (%)")
plt.title("Chance-constraint violations under disturbance (Monte Carlo)")
plt.legend()

# annotate each bar with its value
for bar, rate in zip(bars, df["rate_pct"]):
    plt.text(bar.get_x() + bar.get_width()/2, rate + 0.3,
             f"{rate:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("output/mc_violations.png", dpi=120)
plt.show()
