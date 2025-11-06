# ================================================================"
# MLP Regressor with Few Data Adaptation
# This script is for experiment for injecting new data to adpat the model
# Based on the injection method:
# 1. [Overall Injection] injects t*100% new data into training. Because we have {cell,SOH,SOC} * (n trials) of
#     samples. The method injects t*n samples for each {cell,SOH,SOC} label.
# 2. [Selective Injection] injects t*100% new data into training for selected injection list. Note the injected data
#      will be removed from the testset. This case mimics "Continual Learning".
# ================================================================"
import os
import math
import json
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from MLPModel import *

# ----------------------------
# Global Constants
# ----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# Helper Functions
# ================================================================
def build_cell_colormap(metadata, shade_min=0.30, shade_max=0.90, base_maps=None):
    default_maps = {0: "Blues", 25: "Greens", 45: "Reds"}
    if base_maps:
        default_maps.update(base_maps)

    groups = {}
    for cell, info in metadata.items():
        t = int(info["temperature"])
        groups.setdefault(t, []).append(cell)

    cell_to_color = {}
    for t, cells in groups.items():
        cells_sorted = sorted(cells)
        n = len(cells_sorted)
        positions = [0.6] if n == 1 else np.linspace(shade_min, shade_max, n)
        cmap_name = default_maps.get(t, None)
        cmap = mpl.colormaps.get(cmap_name, mpl.colormaps["gray"])
        for cell, pos in zip(cells_sorted, positions):
            cell_to_color[cell] = mpl.colors.to_hex(cmap(pos))
    return cell_to_color


def build_condition_key(df: pd.DataFrame, soh_round=3, soc_round=1):
    soh_bin = df["SOH"].round(soh_round)
    soc_bin = df["SOC"].round(soc_round)
    key = df["CELL"].astype(str) + "|" + soh_bin.astype(str) + "|" + soc_bin.astype(str)
    return key, soh_bin, soc_bin

### Functional Data loader and Split

def parse_soc_range(expr: str) -> Tuple[float, float, bool, bool]:
    """
    Parse a SOC range expression like "(0.3,0.5]" or "[0.8,1]" into
    (low, high, include_low, include_high).
    """
    expr = expr.strip()
    include_low = expr.startswith("[")
    include_high = expr.endswith("]")
    expr = expr.strip("[]()")
    low, high = [float(x) for x in expr.split(",")]
    return low, high, include_low, include_high


def filter_by_soc(df: pd.DataFrame, soc_ranges: List[str]) -> pd.DataFrame:
    """Filter rows whose SOC falls into *any* of the provided ranges."""
    mask_total = np.zeros(len(df), dtype=bool)
    for expr in soc_ranges:
        low, high, inc_low, inc_high = parse_soc_range(expr)
        mask = (df["SOC"] > low if not inc_low else df["SOC"] >= low) & \
               (df["SOC"] < high if not inc_high else df["SOC"] <= high)
        mask_total |= mask
    return df[mask_total]


def split_with_test_injection(
    df: pd.DataFrame,
    train_cells: list,
    test_cells: list,
    t: float = 0.2,
    features: list = ["R0", "R1", "R2", "R3"],
    target: str = "SOH",
    val_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    Split dataset with optional injection of a fraction of test samples into training.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at least ["CELL", "SOH", "SOC"].
    train_cells : list of str
        Cell names to use for training.
    test_cells : list of str
        Cell names to use for testing.
    t : float
        Fraction (0-1) of test data to include in training.
    features : list of str
        Feature column names.
    target : str
        Target column name.
    val_ratio : float
        Fraction of training data used for validation.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, df_injected
    """
    random.seed(random_state)
    np.random.seed(random_state)

    required_cols = {"CELL", "SOH", "SOC"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must include columns: {required_cols}")

    # Separate train and test pools by cell
    df_train_pool = df[df["CELL"].isin(train_cells)].copy()
    df_test_pool = df[df["CELL"].isin(test_cells)].copy()

    print(f"\n===== Data Split Summary =====")
    print(f"Initial training pool: {len(df_train_pool)} samples from {len(train_cells)} cells ({train_cells})")
    print(f"Initial testing pool : {len(df_test_pool)} samples from {len(test_cells)} cells ({test_cells})")

    # Inject portion of test data into training
    n_inject = int(len(df_test_pool) * t)
    if n_inject > 0:
        df_injected = df_test_pool.sample(n=n_inject, random_state=random_state)
        df_train_pool = pd.concat([df_train_pool, df_injected], ignore_index=True)
        df_test = df_test_pool.drop(df_injected.index).reset_index(drop=True)
    else:
        df_injected = pd.DataFrame(columns=df.columns)
        df_test = df_test_pool.copy().reset_index(drop=True)

    # Split train pool → train/val
    if len(df_train_pool) > 1:
        df_train, df_val = train_test_split(
            df_train_pool, test_size=val_ratio, random_state=random_state, shuffle=True
        )
    else:
        df_train, df_val = df_train_pool.copy(), pd.DataFrame(columns=df.columns)

    # --- Print summaries ---
    def print_group(label, dframe):
        if dframe.empty:
            print(f"\n{label}: [empty]")
            return
        grouped = dframe.groupby(["CELL", "SOH", "SOC"]).size().reset_index(name="count")
        print(f"\n{label} (total {len(dframe)} samples):")
        print(grouped.to_string(index=False))

    print("\n===== Injected Test Samples (added to Training) =====")
    print_group("Injected", df_injected)

    # print("\n===== Final Training/Validation/Test Split =====")
    # print_group("Train", df_train)
    # print_group("Validation", df_val)
    # print_group("Test", df_test)

    # --- Convert to numpy arrays ---
    def to_arrays(dframe):
        if dframe.empty:
            n_feat = len(features)
            return np.zeros((0, n_feat)), np.zeros((0,))
        return (
            dframe[features].values.astype(np.float32),
            dframe[target].values.astype(np.float32),
        )

    X_train, y_train = to_arrays(df_train)
    X_val, y_val = to_arrays(df_val)
    X_test, y_test = to_arrays(df_test)

    print(f"\n===== Summary =====")
    print(f"Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Injected {len(df_injected)} test samples into training ({t*100:.1f}% of test set)")
    print(f"Unique training cells: {sorted(df_train['CELL'].unique())}")
    print(f"Unique testing cells : {sorted(df_test['CELL'].unique())}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, df_injected


def split_with_explicit_injection(
    df: pd.DataFrame,
    train_cells: list,
    test_cells: list,
    injection_cells: list,
    t: float,
    features: list,
    target: str,
    val_ratio: float = 0.2,
    random_state: int = 42
):
    """
    Split data into training, validation, and test sets, with controlled injection
    from a specified subset of test cells.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with columns including CELL, SOH, SOC, and features.
    train_cells : list
        List of cells used for training.
    test_cells : list
        List of all test cells.
    injection_cells : list
        Subset of test_cells from which data can be injected into training.
    t : float
        Fraction of injection cell samples to include into training.
    features : list
        Feature columns.
    target : str
        Target column (e.g., "SOH").
    val_ratio : float
        Fraction of training data reserved for validation.
    random_state : int
        Reproducibility seed.
    """

    # --- Separate main groups ---
    df_train_all = df[df["CELL"].isin(train_cells)].copy()
    df_test_all = df[df["CELL"].isin(test_cells)].copy()
    df_injection_pool = df[df["CELL"].isin(injection_cells)].copy()

    # --- Injection subset selection ---
    if len(df_injection_pool) > 0 and t > 0:
        n_inject = int(len(df_injection_pool) * t)
        df_injected = df_injection_pool.sample(n=n_inject, random_state=random_state)
        df_test_final = df_test_all.drop(df_injected.index, errors="ignore").copy()
    else:
        df_injected = pd.DataFrame(columns=df.columns)
        df_test_final = df_test_all.copy()

    # --- Merge injection into training ---
    df_train_merged = pd.concat([df_train_all, df_injected], ignore_index=True)

    # --- Validation split ---
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(df_train_merged)),
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )

    df_train = df_train_merged.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_merged.iloc[val_idx].reset_index(drop=True)
    df_test = df_test_final.reset_index(drop=True)

    # --- Scale features ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features].values)
    X_val = scaler.transform(df_val[features].values)
    X_test = scaler.transform(df_test[features].values)

    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # --- Print summary ---
    print(f"Train cells: {train_cells}")
    print(f"Test cells: {test_cells}")
    print(f"Injection cells: {injection_cells}")
    print(f"Injected {len(df_injected)} samples ({t*100:.1f}% of injection pool) into training.\n")
    print(f"Train samples: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    if len(df_injected) > 0:
        print("\n===== Injected Samples by (CELL, SOH, SOC) =====")
        grouped = (
            df_injected.groupby(["CELL", "SOH", "SOC"])
            .size()
            .reset_index(name="count")
        )
        for _, row in grouped.iterrows():
            print(f"CELL={row['CELL']}, SOH={row['SOH']:.3f}, SOC={row['SOC']:.3f} → {row['count']} samples")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        df_train, df_val, df_test, df_injected
    )

# ================================================================
# Visualization and Reporting
# ================================================================
def plot_predictions(df, y_true, y_pred, color_map, title="SOH Prediction", save_dir="MLP_plots/leave_one_out_test/select_SOC", title_CELL="", title_dataset=""):
    plot_range = [1.45, 3.75]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c=df["CELL"].map(color_map), s=18, alpha=0.8)
    
    plt.plot(plot_range, plot_range, 'k--', lw=2, label="Ideal = y=x")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(title)
    plt.xlim(plot_range[0], plot_range[1])
    plt.ylim(plot_range[0], plot_range[1])
    plt.grid(True, alpha=0.3)
    handles = [mpatches.Patch(color=color_map[cell], label=cell)
               for cell in sorted(df["CELL"].unique()) if cell in color_map]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    savepath = os.path.join(save_dir, f"{title_CELL}_{title_dataset}.png")
    plt.savefig(savepath, dpi=300)
    plt.close()

def evaluate_and_plot(model, X, y, df_subset, title, color_map, save_path=None, plot_save_dir="MLP_plots/leave_one_out_test/select_SOC", plot_title_CELL="", plot_title_dataset=""):
    """Evaluate model on a given dataset and plot true vs predicted SOH."""
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X).float().to(DEVICE)).squeeze(1).cpu().numpy()

    # --- Metrics ---
    mae = mean_absolute_error(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    mape = np.mean(np.abs((y - y_pred) / np.clip(np.abs(y), 1e-6, None))) * 100.0
    r2 = r2_score(y, y_pred)

    print(f"\n== {title} Results ==")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")

    # --- Build dataframe for plotting ---
    df_vis = df_subset.copy().reset_index(drop=True)
    df_vis["y_true"] = y
    df_vis["y_pred"] = y_pred
    df_vis["error"] = df_vis["y_pred"] - df_vis["y_true"]

    # --- Plot ---
    plot_predictions(df_vis, df_vis["y_true"], df_vis["y_pred"], color_map, title, plot_save_dir, plot_title_CELL, plot_title_dataset)

    # --- Optional save ---
    if save_path:
        df_vis.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df_vis, mae, rmse, mape

# ================================================================
# Main Execution Pipeline
# ================================================================
def main():
    # -------------------------------
    # Load metadata and dataset
    # -------------------------------
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json"
    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)
    color_map = build_cell_colormap(battery_metadata)

    df_all = pd.read_csv("newdf_global_all.csv", index_col=0)
    FEATURES = ["R0", "R1", "R2", "R3", "SOC"]
    TARGET = "SOH"
    df_all = df_all.dropna(subset=["CELL", "SOH", "SOC", "Temp"] + FEATURES).reset_index(drop=True)

    cond_all, soh_bin, soc_bin = build_condition_key(df_all)
    df_all = df_all.assign(COND_ID=cond_all, SOH_BIN=soh_bin, SOC_BIN=soc_bin)

    # -------------------------------
    # Continual Learning Setup
    # -------------------------------
    cells_order = ["CELL090","CELL050","CELL042","CELL076","CELL045","CELL096","CELL013","CELL054"]
    base_cell = cells_order[0]
    all_metrics = []

    # progressively add new cells
    for task_idx in range(1, len(cells_order)): # Last cell added then no test cells left
        train_cells = cells_order[:task_idx]
        unseen_cells = cells_order[task_idx:]  # still unseen
        injection_cells = []  # we’re doing full inclusion at each stage
        t = 1.0

        print(f"\n============================")
        print(f"Continual Learning Stage {task_idx}/{len(cells_order)}")
        print(f"Training on: {train_cells}")
        print(f"Evaluating on unseen cells: {unseen_cells}")
        print(f"============================")

        # Split train/val/test dynamically
        X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, df_injected = \
            split_with_explicit_injection(
                df=df_all,
                train_cells=train_cells,
                test_cells=unseen_cells,
                injection_cells=injection_cells,
                t=t,
                features=FEATURES,
                target=TARGET,
                val_ratio=0.2,
                random_state=42,
            )

        plot_save_dir = f"MLP_plots/CL25_byCELL/task{task_idx}/"
        os.makedirs(plot_save_dir, exist_ok=True)
        save_title_prefix = f"MLP25CL_task{task_idx}"

        # -------------------------------
        # Train model (or fine-tune)
        # -------------------------------
        train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=256, shuffle=False)

        # create or reuse model
        if task_idx == 1:
            model = MLPRegressor(in_dim=len(FEATURES), hidden=[128, 64, 32], p_drop=0.1).to(DEVICE)
        else:
            # Fine-tune from previous stage
            model = model.to(DEVICE)

        model = train(model, train_loader, val_loader, epochs=400)

        # -------------------------------
        # Evaluate on training + unseen
        # -------------------------------
        df_test_vis, test_mae, test_rmse, test_mape = evaluate_and_plot(
            model,
            X_test,
            y_test,
            df_test,
            f"Task{task_idx} SOH Prediction on Unseen Cells",
            color_map,
            save_path=f"{save_title_prefix}_testset.csv",
            plot_save_dir=plot_save_dir,
            plot_title_CELL=f"Task{task_idx}_Temp25",
            plot_title_dataset="unseen",
        )

        # --- Evaluate on validation set (optional but required for combined curve) ---
        df_val_vis, val_mae, val_rmse, val_mape = evaluate_and_plot(
            model,
            X_val,
            y_val,
            df_val,
            f"Task{task_idx} Validation Set",
            color_map,
            save_path=f"{save_title_prefix}_valset.csv",
            plot_save_dir=plot_save_dir,
            plot_title_CELL=f"Task{task_idx}_Temp25",
            plot_title_dataset="val",
        )

        all_metrics.append({
            "Task": task_idx,
            "Train_Cells": ",".join(train_cells),
            "Unseen_Cells": ",".join(unseen_cells),
            "Val_MAE": val_mae,
            "Val_RMSE": val_rmse,
            "Val_MAPE": val_mape,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_MAPE": test_mape,
        })

    # -------------------------------
    # Save experiment summary
    # -------------------------------
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv("Experiment4_ContinualLearning_25C.csv", index=False)
    print("\n Experiment 4 completed. Metrics saved to 'Experiment4_ContinualLearning_25C.csv'")

    # ============================================================
    #  Summary Table
    # ============================================================
    summary_rows = []
    for i, row in enumerate(all_metrics):
        task = i
        if task == 0:
            injected = "– (only train on CELL090)"
        else:
            injected = ", ".join(cells_order[1:task+1])
        summary_rows.append({
            "Task": task,
            "Injected Cells": injected,
            "Test MAPE (%)": row["Test_MAPE"],
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n==================== Summary: Continual Learning Performance ====================")
    print(summary_df.to_string(index=False, formatters={
        "Test MAPE (%)": "{:.2f}".format
    }))
    print("=================================================================================\n")

    # Save formatted table
    summary_df.to_csv("Experiment4_ContinualLearning_25C_Summary.csv", index=False)
    print("📁 Saved summary table to: Experiment4_ContinualLearning_25C_Summary.csv")

    # ============================================================
    #  Compute combined (Validation + Test) MAPE
    # ============================================================
    # optional if you stored val_mape in all_metrics
    val_mapes = [m.get("Val_MAPE", np.nan) for m in all_metrics]
    test_mapes = [m["Test_MAPE"] for m in all_metrics]
    combined_mapes = []
    for v, t in zip(val_mapes, test_mapes):
        if np.isnan(v):
            combined_mapes.append(t)
        else:
            combined_mapes.append((v + t) / 2.0)  # simple average

    # ============================================================
    #  Plot MAPE vs Task (Test + Combined)
    # ============================================================
    plt.figure(figsize=(7, 4.5))
    plt.plot(summary_df["Task"], test_mapes,
             marker='o', linestyle='-', color='tab:blue', label="Test MAPE")
    plt.plot(summary_df["Task"], combined_mapes,
             marker='s', linestyle='--', color='tab:blue', label="Val+Test MAPE")

    plt.xlabel("Task", fontsize=12)
    plt.ylabel("MAPE (%)", fontsize=12)
    plt.title("SOH Prediction: MAPE vs Task (100% Injection Case)", fontsize=13, weight='bold')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("Experiment4_MAPE_vs_Task.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
