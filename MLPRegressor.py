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


def load_and_split_data(
    csv_path: str,
    train_cells: List[str],
    test_cells: List[str],
    split_by_soc: bool = False,
    train_soc_ranges: List[str] = None,
    test_soc_ranges: List[str] = None,
    features: List[str] = ["R0", "R1", "R2", "R3"],
    target: str = "SOH",
    val_ratio: float = 0.2,
    random_state: int = 42,
    scale_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Load, filter, and split the data according to given cell and SOC conditions.

    The validation set is 20% of the training data by default.
    """

    # ---- Load and clean ----
    df = pd.read_csv(csv_path, index_col=0)
    req_cols = ["CELL", "SOH", "SOC", "Temp"] + features
    df = df.dropna(subset=req_cols)

    # ---- Filter by cell ----
    df_train = df[df["CELL"].isin(train_cells)].copy()
    df_test = df[df["CELL"].isin(test_cells)].copy()

    # ---- Optional SOC range filtering ----
    if split_by_soc:
        if train_soc_ranges is None or test_soc_ranges is None:
            raise ValueError("When split_by_soc=True, both train_soc_ranges and test_soc_ranges must be provided.")
        df_train = filter_by_soc(df_train, train_soc_ranges)
        df_test = filter_by_soc(df_test, test_soc_ranges)

    # ---- Prepare data ----
    X_train_full = df_train[features].values.astype(np.float32)
    y_train_full = df_train[target].values.astype(np.float32)

    # Validation = 20% of training set
    X_train, X_val, y_train, y_val, df_train_split, df_val_split = train_test_split(
        X_train_full, y_train_full, df_train, test_size=val_ratio,
        random_state=random_state, shuffle=True
    )

    X_test = df_test[features].values.astype(np.float32)
    y_test = df_test[target].values.astype(np.float32)

    # ---- Scaling ----
    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    print(f"Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train_split, df_val_split, df_test, scaler

# ================================================================
# Visualization and Reporting
# ================================================================
def plot_predictions(df, y_true, y_pred, color_map, title="SOH Prediction", save_dir="MLP_plots/leave_one_out_test/select_SOC", title_CELL="", title_dataset=""):
    plot_range = [1.45, 3.75]
    plt.figure(figsize=(6, 6))
    # # Change color map #
    # unique_cells = df["CELL"].unique()
    # cmap = plt.cm.get_cmap("tab10", len(unique_cells))
    # color_map = {cell: cmap(i) for i, cell in enumerate(unique_cells)}
    # #####

    plt.scatter(y_true, y_pred, c=df["CELL"].map(color_map), s=10, alpha=0.8)
    
    plt.plot(plot_range, plot_range, 'k--', lw=2, label="Ideal = y=x")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(title)
    # plt.xlim(plot_range[0], plot_range[1])
    # plt.ylim(plot_range[0], plot_range[1])
    plt.grid(True, alpha=0.3)
    handles = [mpatches.Patch(color=color_map[cell], label=cell)
               for cell in sorted(df["CELL"].unique()) if cell in color_map]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.axis("square")
    savepath = os.path.join(save_dir, f"allTemp_{title_dataset}.png")
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

    # --- save ---
    if save_path:
        df_vis.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df_vis, mae, rmse, mape


# ================================================================
# Aggregate Middle 50% by Predicted SOH (per CELL, SOH, SOC)
# ================================================================
def compute_group_performance(df_split):
    required_cols = ["CELL", "SOH", "SOC", "y_pred"]
    if not all(c in df_split.columns for c in required_cols):
        raise ValueError(f"Missing required columns in df_split: {required_cols}")

    group_results = []

    for (cell, soh, soc), sub in df_split.groupby(["CELL", "SOH", "SOC"]):
        if len(sub) < 2:
            continue  # skip very small groups

        sub_sorted = sub.sort_values("y_pred").reset_index(drop=True)
        lower = np.percentile(sub_sorted["y_pred"], 25)
        upper = np.percentile(sub_sorted["y_pred"], 75)
        mid = sub_sorted[(sub_sorted["y_pred"] >= lower) & (sub_sorted["y_pred"] <= upper)]

        # Fallback if empty (e.g., identical predictions)
        if mid.empty:
            if len(sub_sorted) >= 2:
                mid = sub_sorted.iloc[[len(sub_sorted)//2 - 1, len(sub_sorted)//2]]
            else:
                mid = sub_sorted.copy()

        group_results.append({
            "CELL": cell,
            "SOH": soh,
            "SOC": soc,
            "SOH_mean": mid["SOH"].mean(),
            "Pred_mean": mid["y_pred"].mean(),
            "Pred_std": mid["y_pred"].std(ddof=0),
            "N_mid": len(mid)
        })

    df_perf = pd.DataFrame(group_results)
    return df_perf


def eval_model(df_split, name, aggregate=False):
    """Evaluate model performance on either raw or aggregated predictions."""
    if aggregate:
        mask = df_split["SOH_mean"].notna() & df_split["Pred_mean"].notna()
        y_true = df_split.loc[mask, "SOH_mean"]
        y_pred = df_split.loc[mask, "Pred_mean"]
    else:
        mask = df_split["SOH"].notna() & df_split["y_pred"].notna()
        y_true = df_split.loc[mask, "SOH"]
        y_pred = df_split.loc[mask, "y_pred"]

    if mask.any():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
        r2 = r2_score(y_true, y_pred)

        print(f"\n=== {name} Performance ===")
        print(f"MAE  = {mae:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAPE = {mape:.2f}%")
        print(f"R2   = {r2:.4f}")
    else:
        print(f"\nNo valid data to evaluate for {name}.")


def plot_group_summary(df_perf, name, color_map=None, save_dir="MLP_plots/allTemp/features_allRC"):
    plt.figure(figsize=(6, 6))
    plot_range = [1.45, 3.75]
    if "CELL" in df_perf.columns and color_map is not None:
        unique_cells = df_perf["CELL"].unique()
        for cell in unique_cells:
            sub = df_perf[df_perf["CELL"] == cell]
            plt.scatter(
                sub["SOH_mean"], sub["Pred_mean"],
                s=10, alpha=0.9, label=cell,
                color=color_map.get(cell, None)
            )
    else:
        plt.scatter(
            df_perf["SOH_mean"], df_perf["Pred_mean"],
            s=10, alpha=0.9, color="tab:blue", label=name
        )
    
    plt.plot(plot_range, plot_range, "r--", label="y=x")

    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"SOH Prediction on {name} Data (MLP)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="CELLs")
    plt.grid(alpha=0.3)
    # plt.xlim(plot_range[0], plot_range[1])
    # plt.ylim(plot_range[0], plot_range[1])
    plt.axis("equal")
    plt.axis("square")
    plt.tight_layout()
    savepath = os.path.join(save_dir, f"allTemp_{name}_agg.png")
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"Saved plot: {savepath}")

# ================================================================
# Main Execution Pipeline
# ================================================================
def main():
    # ---- Load Metadata and Color Map ----
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json"
    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)
    color_map = build_cell_colormap(battery_metadata)

    # ---- Load Data ----
    data_path = Path("fulldf_global_all.csv")
    if data_path is None:
        raise FileNotFoundError("Missing fulldf_global_all.csv.")
    df_all = pd.read_csv(data_path, index_col=0)
    # df_all = df_all[df_all["Temp"] == 25] #NOTE: Select 25 only

    # ---- Clean Data ----
    # FEATURES = ["R0", "R1", "R2", "R3", "SOC"] #NOTE: Remove Temp for single Temp case
    FEATURES = ["R0", "R1", "R2", "R3", "Temp"]

    TARGET = "SOH"
    REQ_COLS = ["CELL", "SOH", "SOC", "tau1", "tau2", "tau3", "Aw", "Temp"] + FEATURES
    df_all = df_all.dropna(subset=REQ_COLS).reset_index(drop=True)
    cond_all, all_sohbin, all_socbin = build_condition_key(df_all)
    df_all = df_all.assign(COND_ID=cond_all, SOH_BIN=all_sohbin, SOC_BIN=all_socbin)

    # ---- Split ----
    ####### ALL DATA ########
    # X_all = df_all[FEATURES].values.astype(np.float32)
    # y_all = df_all[TARGET].values.astype(np.float32)
    # all_idx = np.arange(len(df_all))

    # X_train, X_temp, y_train, y_temp, train_idx, temp_idx = train_test_split(
    #     X_all, y_all, all_idx, test_size=0.2, random_state=42
    # )
    # X_val, X_test, y_val, y_test, val_idx, test_idx = train_test_split(
    #     X_temp, y_temp, temp_idx, test_size=0.5, random_state=42
    # )
    # # create dfs
    # df_train = df_all.iloc[train_idx].reset_index(drop=True)
    # df_val   = df_all.iloc[val_idx].reset_index(drop=True)
    # df_test  = df_all.iloc[test_idx].reset_index(drop=True)
    # print(f"\nTrain samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # plot_save_dir = "MLP_plots/allTemp/Rs_SOC_Temp"
    # if not os.path.exists(plot_save_dir):
    #     os.makedirs(plot_save_dir)
    
     ####### By CELL Split DATA ########
    # train_cells = ["CELL009", "CELL021", "CELL077"] + ["CELL032", "CELL070", "CELL101"] # all 0 & 45
    # test_cells = ["CELL013", "CELL042", "CELL045", "CELL050", "CELL054", "CELL076", "CELL090", "CELL096"] # all 25
    # train_cells = [ "CELL013","CELL045","CELL042","CELL054","CELL076","CELL090", "CELL096"]
    # test_cells = ["CELL050",]
    # train_cells = ["CELL009","CELL021", ]
    # test_cells = ["CELL077"]
    # train_cells = ["CELL032", "CELL070"]
    # test_cells = ["CELL101"]
    train_cells = ["CELL076", "CELL013", "CELL096", "CELL050", "CELL042",  "CELL045", "CELL077", "CELL021", "CELL032", "CELL070"]
    test_cells  = ["CELL090", "CELL054", "CELL009", "CELL101"]  


    train_soc_ranges = []
    test_soc_ranges = []

    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, scaler = load_and_split_data(
        csv_path="fulldf_global_all.csv",
        train_cells=train_cells,
        test_cells=test_cells,
        split_by_soc=False,
        train_soc_ranges=train_soc_ranges,
        test_soc_ranges=test_soc_ranges,
        val_ratio=0.2, 
        features=FEATURES,
        target=TARGET
    )
    plot_save_dir = "MLP_plots/AllTemp_Leaveout/Rs_Temp"
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    # ---- Torch Dataloaders ----
    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=256, shuffle=False)

    # ---- Model ----
    model = MLPRegressor(in_dim=len(FEATURES), hidden=[128, 64, 32], p_drop=0.1).to(DEVICE)

    # ---- Train ----
    model = train(model, train_loader, val_loader, epochs=400)
    train_df_vis, train_mae, train_rmse, train_mape = evaluate_and_plot(
        model,
        X_train,
        y_train,
        df_train,
        "SOH Prediction on Training Data",
        color_map,
        save_path=f"{plot_save_dir}/MLP_predictions_trainset.csv",
        plot_save_dir=plot_save_dir,
        plot_title_CELL="",
        plot_title_dataset="train"
    )

    # ---- Validation Visualization ----
    val_df_vis, val_mae, val_rmse, val_mape = evaluate_and_plot(
        model,
        X_val,
        y_val,
        df_val,
        "SOH Prediction on Validation Data",
        color_map,
        save_path=f"{plot_save_dir}/MLP_predictions_valset.csv",
        plot_save_dir=plot_save_dir,
        plot_title_CELL="",
        plot_title_dataset="val"
    )
    
    # ---- Test Visualization ----
    test_df_vis, test_mae, test_rmse, test_mape = evaluate_and_plot(
        model,
        X_test,
        y_test,
        df_test,
        "SOH Prediction on Test Data",
        color_map,
        save_path=f"{plot_save_dir}/MLP_predictions_testset.csv",
        plot_save_dir=plot_save_dir,
        plot_title_CELL="",
        plot_title_dataset="test"
    )

    # ================================================================
    # Compute and visualize aggregated results
    # ================================================================
    df_val_performance = compute_group_performance(val_df_vis)
    df_test_performance = compute_group_performance(test_df_vis)

    # Save
    df_val_performance.to_csv(f"{plot_save_dir}/MLP_predictions_valset_aggregated.csv", index=False)
    df_test_performance.to_csv(f"{plot_save_dir}/MLP_predictions_testset_aggregated.csv", index=False)

    # Evaluate aggregated performance
    eval_model(df_val_performance, "Validation (Aggregated)", aggregate=True)
    eval_model(df_test_performance, "Test (Aggregated)", aggregate=True)

    # Plot aggregated performance
    plot_group_summary(df_val_performance, "Validation", color_map, save_dir=plot_save_dir)
    plot_group_summary(df_test_performance, "Test", color_map, save_dir=plot_save_dir)

if __name__ == "__main__":
    main()
