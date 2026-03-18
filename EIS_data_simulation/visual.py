import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


mat = sio.loadmat("xy_data_16k_6circuit_v2.mat")

X = mat["x_data"]              # shape: (N, 3, n_freq)
Y = mat["y_data"].flatten()    # shape: (N,)

param_df = pd.read_csv("paramc1-c6_gRange.csv")

num_classes = 6
samples_per_class = 10

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

all_real = []
all_imag = []

selected_indices = []
selected_labels = []


for c in range(num_classes):

    idx = np.where(Y == c)[0]

    chosen = np.random.choice(idx, samples_per_class, replace=False)

    selected_indices.extend(chosen)
    selected_labels.extend([c]*len(chosen))

    ax = axes[c]

    for i in chosen:
        phase_deg = X[i, 1, :]
        mag = X[i, 2, :]

        theta = np.deg2rad(phase_deg)
        theta = np.unwrap(theta)

        Z_real = mag * np.cos(theta)
        Z_imag = mag * np.sin(theta)

        ax.plot(Z_real, -Z_imag, alpha=0.7)

        all_real.extend(Z_real)
        all_imag.extend(-Z_imag)

    ax.set_title(f"Circuit {c+1}")
    ax.set_xlabel("Z' (Ohm)")
    ax.set_ylabel("-Z'' (Ohm)")
    ax.grid(True)



xmin, xmax = min(all_real), max(all_real)
ymin, ymax = min(all_imag), max(all_imag)

for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()


subset = param_df.iloc[selected_indices].copy()
subset["CircuitLabel"] = selected_labels

subset.to_csv("selected60_for_visual_check.csv", index=False)

print("Finished.")
print("Generated:")
print("- Unified-scale Nyquist plot")
print("- selected60_for_visual_check.csv")
