import os
import math
import json
import numpy as np
import pandas as pd
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
def plot_predictions(df, y_true, y_pred, color_map, title="SOH Prediction"):
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
    plt.show()

def evaluate_and_plot(model, X, y, df_subset, title, color_map, save_path=None):
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
    plot_predictions(df_vis, df_vis["y_true"], df_vis["y_pred"], color_map, title)

    # --- Optional save ---
    if save_path:
        df_vis.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    return df_vis, mae, rmse, mape

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

    # ---- Clean Data ----
    FEATURES = ["R0", "R1", "R2", "R3", "SOC", "Temp"]
    TARGET = "SOH"
    REQ_COLS = ["CELL", "SOH", "SOC", "Temp"] + FEATURES
    df_all = df_all.dropna(subset=REQ_COLS).reset_index(drop=True)
    cond_all, all_sohbin, all_socbin = build_condition_key(df_all)
    df_all = df_all.assign(COND_ID=cond_all, SOH_BIN=all_sohbin, SOC_BIN=all_socbin)

    # ---- Split ----
    # X_all = df_all[FEATURES].values.astype(np.float32)
    # y_all = df_all[TARGET].values.astype(np.float32)
    # all_idx = np.arange(len(df_all))

    # X_train, X_temp, y_train, y_temp, train_idx, temp_idx = train_test_split(
    #     X_all, y_all, all_idx, test_size=0.2, random_state=42
    # )
    # X_val, X_test, y_val, y_test, val_idx, test_idx = train_test_split(
    #     X_temp, y_temp, temp_idx, test_size=0.5, random_state=42
    # )
    
    # train_cells = ["CELL009", "CELL021", "CELL077"] + ["CELL032", "CELL070", "CELL101"] # all 0 & 45
    # test_cells = ["CELL013", "CELL042", "CELL045", "CELL050", "CELL054", "CELL076", "CELL090", "CELL096"] # all 25
    train_cells = ["CELL013","CELL045", "CELL050", "CELL054", "CELL076","CELL090", "CELL096"]
    test_cells = ["CELL042"]
    # train_cells = ["CELL021", "CELL077"]
    # test_cells = ["CELL009"]
    # train_cells = ["CELL070", "CELL101"]
    # test_cells = ["CELL032"]


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
        save_path="MLP_predictions_trainset.csv"
    )


    # ---- Validation Visualization ----
    val_df_vis, val_mae, val_rmse, val_mape = evaluate_and_plot(
        model,
        X_val,
        y_val,
        df_val,
        "SOH Prediction on Validation Data",
        color_map,
        save_path="MLP_predictions_valset.csv"
    )


    # # ---- Test ----
    # with torch.no_grad():
    #     y_pred_test = model(torch.from_numpy(X_test).float().to(DEVICE)).squeeze(1).cpu().numpy()

    # mae = mean_absolute_error(y_test, y_pred_test)
    # rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))
    # mape = np.mean(np.abs((y_test - y_pred_test) / np.clip(np.abs(y_test), 1e-6, None))) * 100.0
    # print(f"\n== Test Set Results ==")
    # print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

    # # ---- Plot Results ----
    # pred_df = df_test.copy().reset_index(drop=True)
    # pred_df["y_true_test"] = y_test
    # pred_df["y_pred_test"] = y_pred_test
    # plot_predictions(pred_df, y_test, y_pred_test, color_map, title="SOH Prediction on Test Set")

    # # ---- Save Artifacts ----
    # pred_df.to_csv("predictions_testset.csv", index=False)
    # print("Saved predictions_testset.csv")
    
    # ---- Test Visualization ----
    test_df_vis, test_mae, test_rmse, test_mape = evaluate_and_plot(
        model,
        X_test,
        y_test,
        df_test,
        "SOH Prediction on Test Data",
        color_map,
        save_path="MLP_predictions_testset.csv"
    )


if __name__ == "__main__":
    main()
