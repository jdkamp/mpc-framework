import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DT = 0.1        # must match config.dt in examples/agent_avoidance/main.cpp
R_SAFE = 1.0    # must match r_safe in main()
T_CUT = 2.0     # must match AgentSim::t_cut (step 20) * DT

det = pd.read_csv("output/agent_deterministic.csv")
cha = pd.read_csv("output/agent_chance.csv")

t = det["t"] * DT
agent_py = det["agent_py"]
boundary = det["clearance"]          # agent_py - r_safe: the ego must stay below this

margin_det = agent_py - R_SAFE - det["py"]
margin_cha = agent_py - R_SAFE - cha["py"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})

# top: lateral positions
ax1.fill_between(t, boundary, 3.0, color="tab:red", alpha=0.12,
                 label=f"inside $r_{{safe}}$ = {R_SAFE} m")
ax1.plot(t, agent_py, color="dimgray", ls="--", lw=1.2, label="agent $p_y$")
ax1.plot(t, boundary, color="tab:red", lw=1.6, label="collision boundary")
ax1.plot(t, det["py"], color="tab:orange", lw=1.8, label="deterministic prediction")
ax1.plot(t, cha["py"], color="tab:green", lw=1.8, label="chance constrained")
ax1.axhline(-2.5, color="black", ls=":", lw=1.0, label="lane bound")

ax1.axvline(T_CUT, color="tab:blue", ls="-.", lw=1.0)
ax1.annotate("cut-in starts", xy=(T_CUT, 2.0), xytext=(6, 0),
             textcoords="offset points", color="tab:blue", fontsize=9)

ax1.set_ylabel("lateral position $p_y$ [m]")
ax1.set_ylim(-2.6, 2.6)
ax1.set_title("Agent cuts in at an unannounced moment: only the chance-constrained\n"
              "controller holds margin for the onset it cannot predict", fontsize=10)
ax1.legend(loc="lower left", fontsize=8, ncol=2)

# bottom: clearance margin
ax2.axhspan(-1.0, 0.0, color="tab:red", alpha=0.12)
ax2.axhline(0.0, color="tab:red", lw=1.4)
ax2.plot(t, margin_det, color="tab:orange", lw=1.8,
         label=f"deterministic (min {margin_det.min():.2f} m)")
ax2.plot(t, margin_cha, color="tab:green", lw=1.8,
         label=f"chance constrained (min {margin_cha.min():.2f} m)")

ax2.axvline(T_CUT, color="tab:blue", ls="-.", lw=1.0)

ax2.set_xlabel("time [s]")
ax2.set_ylabel("margin [m]")
ax2.set_ylim(-0.6, max(margin_det.max(), margin_cha.max()) * 1.1)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("output/agent_avoidance.png", dpi=150, bbox_inches="tight")
plt.show()
