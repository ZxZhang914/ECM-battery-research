# train_on_median_test_on_alltrials.py
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 0) Reproducibility & device
# ----------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 1) Load data
# ----------------------------
import matplotlib as mpl
def build_cell_colormap(metadata, shade_min=0.30, shade_max=0.90, base_maps=None):
    """
    Create a dict: {cell_name: '#RRGGBB'}.
    
    Parameters
    ----------
    metadata : dict
        loaded from json file
    shade_min, shade_max : float in [0,1]
        Range within the colormap to sample shades (avoid extremes that are too light/dark).
    base_maps : dict or None
        Optional override of base colormaps per temperature, e.g. {0:"Blues", 25:"Greens", 45:"Reds"}.
    """
    # Default temperature → base colormap
    default_maps = {0: "Blues", 25: "Greens", 45: "Reds"}
    if base_maps:
        default_maps.update(base_maps)
    
    # Group cells by temperature
    groups = {}
    for cell, info in metadata.items():
        t = int(info["temperature"])
        groups.setdefault(t, []).append(cell)
    
    # Build color map per group, assigning distinct shades
    cell_to_color = {}
    for t, cells in groups.items():
        cells_sorted = sorted(cells)  # deterministic assignment
        n = len(cells_sorted)
        if n == 1:
            positions = [0.6]
        else:
            positions = np.linspace(shade_min, shade_max, n)
        
        cmap_name = default_maps.get(t, None)
        if cmap_name is None or cmap_name not in mpl.colormaps:
            # Fallback if an unexpected temperature shows up
            cmap = mpl.colormaps["gray"]
        else:
            cmap = mpl.colormaps[cmap_name]
        
        for cell, pos in zip(cells_sorted, positions):
            rgba = cmap(pos)
            cell_to_color[cell] = mpl.colors.to_hex(rgba)
    
    return cell_to_color

battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json" # Check the path

with open(battery_json_file, "r") as f:
    battery_metadata = json.load(f)   # <--- this is now a dict

COLOR_MAP = build_cell_colormap(battery_metadata)

# Load Data

alltrials_path_candidates = ["df_global_all.csv"] 

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

alltrials_path = first_existing(alltrials_path_candidates)


if alltrials_path is None:
    raise FileNotFoundError("df_global_all.csv (or df_gloabl_all.csv) not found.")

df_all = pd.read_csv(alltrials_path, index_col=0)
df_all["Temp"] = pd.to_numeric(df_all["Temp"], errors='coerce')
df_all["1overTemp"] = 1.0 / (df_all["Temp"]+273.15)
# ----------------------------
# 2) Columns & cleaning
# ----------------------------
FEATURES = ["R0","R1","R2","R3","SOC","Temp"]
TARGET = "SOH"
REQ_COLS = ["CELL","SOH","SOC","Temp","R0","R1","R2","R3"]

for col in REQ_COLS:
    if col not in df_all.columns:
        raise ValueError(f"All-trials df missing column: {col}")

# Drop NA rows for the used columns
df_all = df_all.dropna(subset=REQ_COLS).reset_index(drop=True)

# # Keep SOH within sane bounds, adjust if needed
# df_med[TARGET] = df_med[TARGET].clip(lower=0.0, upper=1.2)
# df_all[TARGET] = df_all[TARGET].clip(lower=0.0, upper=1.2)

# # 0) Basic overview
# print("Cells in ALL-TRIALS (raw):", df_all["CELL"].nunique())
# print(df_all["CELL"].value_counts())

# # 1) Check missing by column and by cell
# print("\nMissing per column in ALL-TRIALS:")
# print(df_all[["CELL","SOH","SOC","Temp","R0","R1","R2","R3"]].isna().sum())

# print("\nRows per cell BEFORE dropna:")
# print(df_all.groupby("CELL").size().sort_values())

# # 2) If you did dropna earlier, check AFTER dropna
# df_all_after = df_all.dropna(subset=["SOH","SOC","R0","R1","R2","R3","Temp"])
# print("\nRows per cell AFTER dropna:")
# print(df_all_after.groupby("CELL").size().sort_values())

# missing_cells = sorted(set(df_all["CELL"]) - set(df_all_after["CELL"]))
# print("\nCells eliminated by dropna:", missing_cells)


# ----------------------------
# 3) Build "condition id" to keep identical operating points together
# ----------------------------
def build_condition_key(df: pd.DataFrame, soh_round=3, soc_round=1):
    soh_bin = df["SOH"].round(soh_round)
    soc_bin = df["SOC"].round(soc_round)
    key = df["CELL"].astype(str) + "|" + soh_bin.astype(str) + "|" + soc_bin.astype(str)
    return key, soh_bin, soc_bin


cond_all, all_sohbin, all_socbin = build_condition_key(df_all, soh_round=3, soc_round=1)
df_all = df_all.assign(COND_ID=cond_all, SOH_BIN=all_sohbin, SOC_BIN=all_socbin)


# ----------------------------
# 4) Split: train/test with t% of 25°C in training
# ----------------------------
t = 0  # fraction of 25°C data used for training (0.0 ≤ t ≤ 1.0)

df_045 = df_all[df_all["Temp"].isin([45, 0])]
# df_25  = df_all[df_all["Temp"] == 25].copy()
df_25 = df_all[df_all["Temp"].isin([25])]

if t >= 1.0:
    print("t=1 results in no test data. For evaluation, use t < 1.0.")
    t = 0.9

if t <= 0.0:
    # No 25°C data in training
    df_25_train = pd.DataFrame(columns=df_25.columns)
    df_25_test = df_25
else:
    # Split normally
    df_25_train, df_25_test = train_test_split(
        df_25, test_size=(1 - t), random_state=42, shuffle=True
    )

print(len(df_25_train))
# Combine
# df_train = pd.concat([df_045, df_25_train], ignore_index=True)
df_train = df_045
df_test  = df_25_test.reset_index(drop=True)

# Prepare arrays
X_train = df_train[FEATURES].values.astype(np.float32)
print(X_train.shape)
y_train = df_train[TARGET].values.astype(np.float32)
X_test  = df_test[FEATURES].values.astype(np.float32) if len(df_test) > 0 else np.array([]).reshape(0, len(FEATURES))
y_test  = df_test[TARGET].values.astype(np.float32) if len(df_test) > 0 else np.array([])

print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
print(f"From 25°C: {len(df_25_train)} in training, {len(df_25_test)} in test")

# ----------------------------
# 5) Scale (fit on train only)
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 6) Torch dataset & loaders
# ----------------------------
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)

# ----------------------------
# 7) Model
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64], p_drop: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p_drop)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = MLPRegressor(in_dim=len(FEATURES), hidden=[128, 64, 32], p_drop=0.1).to(DEVICE)

# ----------------------------
# 8) Training setup
# ----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=8)

# ----------------------------
# 9) Training loop (no early stop)
# ----------------------------
def train_model(model, train_loader, epochs=300):
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            # scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)

        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | Train RMSE: {math.sqrt(train_loss):.4f}")

train_model(model, train_loader, epochs=200)
# ----------------------------
# 10) Evaluate on training set
# ----------------------------
# with torch.no_grad():
#     y_pred_train = (model(torch.from_numpy(X_train).float().to(DEVICE))
#                     .squeeze(1).cpu().numpy())

#     train_df_vis = df_train.copy().reset_index(drop=True)
#     train_df_vis["y_true_train"] = y_train
#     train_df_vis["y_pred_train"] = y_pred_train
#     train_df_vis["color"] = train_df_vis["CELL"].astype(str).map(COLOR_MAP)

#     plt.figure(figsize=(6,6))
#     plt.scatter(y_train, y_pred_train,
#                 c=train_df_vis["color"], s=18, alpha=0.8)
#     lims = [min(train_df_vis["y_true_train"].min(), train_df_vis["y_pred_train"].min()),
#             max(train_df_vis["y_true_train"].max(), train_df_vis["y_pred_train"].max())]
#     plt.plot(lims, lims, 'k--', lw=2, label="Ideal = y=x")

#     plt.xlabel("True SOH (train)")
#     plt.ylabel("Predicted SOH (train)")
#     plt.title(f"SOH Prediction on Training Data \n Training with 0°C & 45°C & {t*100}% of 25°C Data")
#     plt.grid(True, alpha=0.3)

#     handles = [mpatches.Patch(color=COLOR_MAP[cell], label=cell)
#             for cell in sorted(train_df_vis["CELL"].unique()) if cell in COLOR_MAP]
#     plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.tight_layout()
#     plt.show()
#     # plt.close()

#     print(y_train.shape, y_pred_train.shape)
#     r2_train = r2_score(y_train, y_pred_train)

#     print(f"Train R²: {r2_train:.4f}")

with torch.no_grad():
    y_pred_train = (
        model(torch.from_numpy(X_train).float().to(DEVICE))
        .squeeze(1)
        .cpu()
        .numpy()
    )

    train_df_vis = df_train.copy().reset_index(drop=True)
    train_df_vis["color"] = train_df_vis["CELL"].astype(str).map(COLOR_MAP)

    plt.figure(figsize=(6, 6))

    n_plot = 1000
    idx = np.random.choice(len(y_train), n_plot)

    plt.scatter(y_train[idx], y_pred_train[idx], c=train_df_vis["color"][idx], s=18, alpha=0.8)

    plt.plot([1, 3.8], [1, 3.8], 'k--', lw=2, label="Ideal = y=x")

    plt.xlabel("True SOH (train)")
    plt.ylabel("Predicted SOH (train)")
    plt.title("SOH Prediction on Training Data")

    plt.grid(True, alpha=0.3)
    handles = [
        mpatches.Patch(color=COLOR_MAP[cell], label=cell)
        for cell in sorted(train_df_vis["CELL"].unique())
        if cell in COLOR_MAP
    ]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.xlim(1.0,3.8)
    plt.ylim(1.0,3.8)
    plt.show()
    plt.close()

print((y_train - y_pred_train).mean())
print((y_train - y_pred_train).max())
print((y_train - y_pred_train).min())

plt.hist(y_train - y_pred_train, bins=100)

# plt.scatter(y_train, y_pred_train)
plt.show()
r2_train = r2_score(y_train, y_pred_train)
print(f"Train R²: {r2_train:.4f}")

# ----------------------------
# 11) Evaluate on test set (if available)
# ----------------------------
if len(X_test) > 0:
    with torch.no_grad():
        y_pred_test = (model(torch.from_numpy(X_test).float().to(DEVICE))
                      .squeeze(1).cpu().numpy())

    overall_mae  = mean_absolute_error(y_test, y_pred_test)
    overall_rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))
    overall_mape = np.mean(np.abs((y_test - y_pred_test) / np.clip(np.abs(y_test), 1e-6, None))) * 100.0

    print("\n== Test Set Results ==")
    print(f"MAE : {overall_mae:.5f} | RMSE: {overall_rmse:.5f} | MAPE: {overall_mape:.2f}%")

    r2_test = r2_score(y_test, y_pred_test)
    print(f"Test R²: {r2_test:.4f}")

    # ----------------------------
    # 12) Save predictions & grouped reports
    # ----------------------------
    pred_df = df_test.copy().reset_index(drop=True)
    pred_df["y_true_test"] = y_test
    pred_df["y_pred_test"] = y_pred_test

    # Compute errors
    pred_df["err"] = pred_df["y_pred_test"] - pred_df["y_true_test"]
    pred_df["abs_err"] = pred_df["err"].abs()
    pred_df["rel_err"] = pred_df["err"] / pred_df["y_true_test"].replace(0, np.nan)
    pred_df["color"] = pred_df["CELL"].map(COLOR_MAP)

    # Per-condition summary
    grp_cols = ["CELL","SOH_BIN","SOC_BIN","COND_ID"]
    cond_report = (pred_df.groupby(grp_cols)
                   .agg(
                       n_trials=("y_true_test","count"),
                       true_mean=("y_true_test","mean"),
                       pred_mean=("y_pred_test","mean"),
                       err_mean=("err","mean"),
                       err_std=("err","std"),
                       abs_err_mean=("abs_err","mean"),
                       abs_err_std=("abs_err","std"),
                   )
                   .reset_index())

    # Per-cell summary
    cell_report = (pred_df.groupby("CELL")
                   .agg(
                       n_rows=("y_true_test","count"),
                       mae=("abs_err","mean"),
                       rmse=("err", lambda s: math.sqrt(np.mean(s**2))),
                       mape=("err", lambda s: float(np.mean(np.abs(s / np.clip(pred_df.loc[s.index, 'y_true_test'].values, 1e-6, None))) * 100.0))
                   )
                   .reset_index())

    pred_df.to_csv("predictions_testset.csv", index=False)
    cond_report.to_csv("report_per_condition_testset.csv", index=False)
    cell_report.to_csv("report_per_cell_testset.csv", index=False)
    print("Saved: predictions_testset.csv, report_per_condition_testset.csv, report_per_cell_testset.csv")

    # ----------------------------
    # 13) Visualization (Test Results)
    # ----------------------------
    # 1. Scatter: True vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(pred_df["y_true_test"], pred_df["y_pred_test"],
                c=pred_df["color"], s=18, alpha=0.55)
    lims = [min(pred_df["y_true_test"].min(), pred_df["y_pred_test"].min()),
            max(pred_df["y_true_test"].max(), pred_df["y_pred_test"].max())]
    plt.plot(lims, lims, 'k--', lw=2, label="Ideal = y=x")
    plt.xlabel("True SOH (test)")
    plt.ylabel("Predicted SOH (test)")
    plt.title(f"SOH Prediction on Testing Data (t={t:.2f}) \n Training with 0°C & 45°C & {t*100}% of 25°C Data")
    handles = [mpatches.Patch(color=COLOR_MAP[cell], label=cell)
               for cell in sorted(pred_df["CELL"].unique()) if cell in COLOR_MAP]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Error histogram
    errors = y_pred_test - y_test
    plt.figure(figsize=(7,4))
    plt.hist(errors, bins=50, color="steelblue", edgecolor="k", alpha=0.7)
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Prediction Error (Pred - True) Testset")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Prediction Errors on Testset (t={t:.2f})")
    plt.tight_layout()
    plt.show()

    # 3. Per-cell error distributions
    plt.figure(figsize=(10,5))
    sns.boxplot(data=pred_df, x="CELL", y="err", showfliers=False, palette=COLOR_MAP)
    plt.axhline(0, color="k", linestyle="--")
    plt.ylabel("Error (Pred - True)")
    plt.title(f"Error Distribution per Cell (t={t:.2f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    sns.boxplot(data=pred_df, x="CELL", y="rel_err", showfliers=False, palette=COLOR_MAP)
    plt.axhline(0, color="k", linestyle="--")
    plt.ylabel("Relative Error: (Pred - True) / True")
    plt.title(f"Relative Error per Cell (t={t:.2f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

else:
    print("\nNo test data available (t=1.0). Skipping test evaluation and plots.")