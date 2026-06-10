import numpy as np
import pandas as pd

# Parameters
L = 2.7             # wheelbase (m)
k = 0.004           # disturbance gain
n_samples = 200     # number of training samples

np.random.seed(42)  # for reproducibility

# Sample (v, delta)
# Most samples should cluster arounf "typical" driving conditions
#    - lower velocities (5 m/s)
#    - small steering angles (centered around 0)
# with spaprse coverage towards the limits (~15 m/s, +/-0.5rad/s)

v = np.random.normal(loc=5, scale=2, size=n_samples)  # mean 5 m/s, std 2 m/s
v = np.clip(v, 0, 30)  # limit to [0, 30] m/s
delta = np.random.normal(loc=0, scale=0.2, size=n_samples)  # mean 0 rad, std 0.2 rad
delta = np.clip(delta, -0.5, 0.5)  # limit to [-0.5, 0.5] rad


# Compute the "true" residual + noise
# residual = -k * v^2 * delta + noise

noise = np.random.normal(loc=0, scale=0.005, size=n_samples)  # mean 0, std 0.005
residual = -k * v**2 * delta + noise

# Save to CSV
df = pd.DataFrame({
    'v': v,
    'delta': delta,
    'residual': residual
})
df.to_csv('data/gp_training_data.csv', index=False)

print(df.describe())

