import numpy as np
import pandas as pd
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('data/gp_training_data.csv')
X = df[['v', 'delta']].values
y = df['residual'].values

# Define and fill GP
kernel = (ConstantKernel(0.001, constant_value_bounds=(1e-5, 0.1)) *
          RBF(length_scale=[3.0, 0.2], length_scale_bounds=[(0.5, 8.0), (0.05, 0.5)]))
noise = 1e-5
gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise, n_restarts_optimizer=5)
gpr.fit(X, y)
print("Learned kernel: ", gpr.kernel_)

# Extract kernel hyperparameters
k = gpr.kernel_
sigma_f_sq = float(k.k1.constant_value)
length_scale = k.k2.length_scale.tolist()

# Export GP parameters
params = {
    'kernel': {
        "output_scale": sigma_f_sq,
        "length_scale": length_scale,
        "noise_variance": noise
    },
    "X_train": X.tolist(),
    "y_train": y.tolist(),
}
with open('data/gp_params.json', 'w') as f:
    json.dump(params, f)

print("GP parameters exported to gp_params.json")

# Reference predictions for testing
test_points = np.array([[5.0, 0.1], [10.0, 0.3], [3.0, -0.2], [8.0, 0.0]])
ref_means, ref_stds = gpr.predict(test_points, return_std=True)
print("\n--- Test reference values ---")
for (v, d), m, s in zip(test_points, ref_means, ref_stds):
    print(f"v={v:>5}, delta={d:>5}:  mean={m: .6f}  var={s**2: .6f}")

# Plot mean and variance
v_grid = np.linspace(0, 15, 50)
d_grid = np.linspace(-0.5, 0.5, 50)
VV, DD = np.meshgrid(v_grid, d_grid)
X_grid = np.column_stack([VV.ravel(), DD.ravel()])
mean, std = gpr.predict(X_grid, return_std=True)
mean = mean.reshape(50, 50)
var  = (std**2).reshape(50, 50)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
c1 = axes[0].contourf(VV, DD, mean, levels=30, cmap='RdBu')
plt.colorbar(c1, ax=axes[0])
axes[0].set_title('GP Posterior Mean')
axes[0].set_xlabel('v [m/s]')
axes[0].set_ylabel('delta [rad]')
c2 = axes[1].contourf(VV, DD, var, levels=30, cmap='YlOrRd')
plt.colorbar(c2, ax=axes[1])
axes[1].set_title('GP Posterior Variance')
axes[1].set_xlabel('v [m/s]')
axes[1].set_ylabel('delta [rad]')
plt.tight_layout()
plt.savefig('output/gp_surfaces.png', dpi=150)
plt.show()
print("Saved gp_surfaces.png")
