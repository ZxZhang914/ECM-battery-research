# Leave N out
import os
import json
import math
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import sys
sys.stdout.reconfigure(encoding='utf-8')

from MLPModel import *

# =====================================================
# Config & Random Seed
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# Helper Functions
# =====================================================
def build_cell_colormap(metadata, shade_min=0.30, shade_max=0.90):
    default_maps = {0: "Blues", 25: "Greens", 45: "Reds"}
    groups = {}
    for cell, info in metadata.items():
        t = int(info["temperature"])
        groups.setdefault(t, []).append(cell)

    cell_to_color = {}
    for t, cells in groups.items():
        cells_sorted = sorted(cells)
        n = len(cells_sorted)
        positions = [0.6] if n == 1 else np.linspace(shade_min, shade_max, n)
        cmap_name = default_maps.get(t, "gray")
        cmap = mpl.colormaps.get(cmap_name)
        for cell, pos in zip(cells_sorted, positions):
            cell_to_color[cell] = mpl.colors.to_hex(cmap(pos))
    return cell_to_color


def reduce_training_samples(df_train, cell_col="CELL", soc_col="SOC", soh_col="SOH", random_state=42):
    """Reduced training condition (Experiment 3 logic)."""
    np.random.seed(random_state)
    soc_ranges = [(0.25, 0.5), (0.5, 0.8), (0.8, 1.0)]
    selected_soc_values = []

    for cell, cell_group in df_train.groupby(cell_col):
        for soh, soh_group in cell_group.groupby(soh_col):
            for (low, high) in soc_ranges:
                subset = soh_group[(soh_group[soc_col] >= low) & (soh_group[soc_col] < high)]
                if not subset.empty:
                    soc_value = subset[soc_col].sample(1, random_state=random_state).iloc[0]
                    selected_soc_values.append((cell, soh, soc_value))

    selected_rows = []
    for (cell, soh, soc_value) in selected_soc_values:
        mask = (
            (df_train[cell_col] == cell)
            & (df_train[soh_col] == soh)
            & (np.isclose(df_train[soc_col], soc_value, atol=1e-6))
        )
        selected_rows.append(df_train[mask])

    reduced_df = pd.concat(selected_rows, ignore_index=True)
    print(f"Reduced training samples: {len(reduced_df)} ({len(reduced_df) / len(df_train) * 100:.1f}% of original)")
    return reduced_df


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X).float().to(DEVICE)).squeeze(1).cpu().numpy()

    mae = mean_absolute_error(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    mape = np.mean(np.abs((y - y_pred) / np.clip(np.abs(y), 1e-6, None))) * 100
    r2 = r2_score(y, y_pred)
    return mae, rmse, mape, r2, y_pred


# =====================================================
# Leave-N-Out Implementation
# =====================================================
def leave_n_out(df_all, n, features, target, color_map, reduced_training=True):
    """
    Leave-N-Out cross-validation.
    Trains on all but n cells and tests on n cells.
    """
    all_cells = sorted(df_all["CELL"].unique())
    combs = list(itertools.combinations(all_cells, n))
    print(f"\n* Running Leave-{n}-Out ({len(combs)} combinations)")

    results = []

    for test_cells in combs:
        test_cells = list(test_cells)
        print(f"\n{'='*60}\n** Leave-{n}-Out Test Cells: {test_cells}")

        df_test = df_all[df_all["CELL"].isin(test_cells)].copy()
        df_trainval = df_all[~df_all["CELL"].isin(test_cells)].copy()

        # Split 80/20 for validation
        df_train, df_val = train_test_split(df_trainval, test_size=0.2, random_state=SEED, shuffle=True)

        if reduced_training:
            df_train = reduce_training_samples(df_train)

        X_train = df_train[features].values.astype(np.float32)
        y_train = df_train[target].values.astype(np.float32)
        X_val = df_val[features].values.astype(np.float32)
        y_val = df_val[target].values.astype(np.float32)
        X_test = df_test[features].values.astype(np.float32)
        y_test = df_test[target].values.astype(np.float32)

        # Standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # DataLoaders
        train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=256, shuffle=False)

        # Model #NOTE: May change architecture
        model = MLPRegressor(in_dim=len(features), hidden=[64, 32], p_drop=0.1).to(DEVICE)
        model = train(model, train_loader, val_loader, epochs=400)

        # Evaluate
        test_mae, test_rmse, test_mape, test_r2, y_pred = evaluate_model(model, X_test, y_test)
        print(f"Test — MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%, R2={test_r2:.4f}")

        for cell in test_cells:
            mask = df_test["CELL"] == cell
            if mask.any():
                mae, rmse, mape, r2, _ = evaluate_model(model, X_test[mask], y_test[mask])
                results.append({
                    "n": n,
                    "test_cell": cell,
                    "combo": "_".join(test_cells),
                    "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2
                })

    return pd.DataFrame(results)


# =====================================================
# Plotting Function
# =====================================================
def plot_leave_n_out_results(results_df, save_dir="MLP_plots/LeaveNOut"):
    os.makedirs(save_dir, exist_ok=True)
    cells = sorted(results_df["test_cell"].unique())
    n_values = sorted(results_df["n"].unique())

    plt.figure(figsize=(7, 5))
    for cell in cells:
        df_cell = results_df[results_df["test_cell"] == cell]
        mean_r2 = [df_cell[df_cell["n"] == n]["MAPE"].mean() for n in n_values]
        plt.plot(n_values, mean_r2, marker="o", label=cell, alpha=0.6)

    # Average curve
    avg_r2 = [results_df[results_df["n"] == n]["MAPE"].mean() for n in n_values]
    plt.plot(n_values, avg_r2, "k--", marker="s", linewidth=2.5, label="Average")

    # Make x-axis discrete
    plt.xticks(n_values, [str(int(n)) for n in n_values])
    plt.xlim(min(n_values) - 0.2, max(n_values) + 0.2)

    plt.xlabel("Number of Test Cells (n)")
    plt.ylabel("MAPE (%)")
    plt.title("Leave-N-Out Test Performance")
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "LeaveNOut_MAPE_summary.png"), dpi=300)
    plt.close()


# =====================================================
# Main
# =====================================================
def main():
    # Load data
    df_all = pd.read_csv("fulldf_global_all.csv", index_col=0)
    df_all = df_all[df_all["Temp"] == 25]  # Use only 25°C
    FEATURES = ["R0", "R1", "R2", "R3", "SOC"]
    TARGET = "SOH"

    df_all = df_all.dropna(subset=["CELL", "SOH", "SOC", "Temp"] + FEATURES)
    print(f"Detected {df_all['CELL'].nunique()} cells.")

    # Color map
    with open("../EVC_EIS_Data/original_data/Battery_Info_DRT.json") as f:
        metadata = json.load(f)
    color_map = build_cell_colormap(metadata)

    # Save dir
    save_dir = f"MLP_plots/LeaveNOut"
    os.makedirs(save_dir, exist_ok=True)

    # Run Leave-N-Out for n = 3, 2, 1
    N_values = [7,6,5,4,3,2,1]
    all_results = []
    for n in N_values:
        df_n = leave_n_out(df_all, n, FEATURES, TARGET, color_map, reduced_training=True)
        all_results.append(df_n)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(f"{save_dir}/MLP_LOON_results_all.csv", index=False)
    print("\nSaved: MLP_LOON_results_all.csv")

    # Plot
    plot_leave_n_out_results(results_df, save_dir)
    print("Saved plot: LeaveNOut_MAPE_summary.png")


if __name__ == "__main__":
    main()
    # partial_df = pd.read_csv("MLP_plots/LeaveNOut/MLP_LOON_results_all.csv")
    # partial_df = partial_df[partial_df["n"].isin([1,2,3,4,5,6,7])]
    # plot_leave_n_out_results(partial_df, save_dir="MLP_plots/LeaveNOut")
