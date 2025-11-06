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

def split_by_soc_one_selection_else_to_test(
    df: pd.DataFrame,
    train_cells: List[str],
    train_soc_ranges: List[Tuple[float, float]],
    features: List[str] = ["R0", "R1", "R2", "R3"],
    target: str = "SOH",
    val_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    For each training cell and each SOH:
      - For each SOC range, randomly select ONE SOC (if any exist within that range)
      - Add all samples for (CELL, SOH, chosen SOC) to training set
    Unselected SOCs and all other cells become the test set.
    Split training data 80/20 into train/validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ["CELL", "SOH", "SOC"] at least.
    train_cells : list of str
        List of cell names to use for training.
    train_soc_ranges : list of (float, float)
        SOC ranges to consider for possible selection.
    features : list of str
        Feature column names.
    target : str
        Target column name.
    val_ratio : float
        Fraction of training data for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, selection_df
    """
    random.seed(random_state)
    np.random.seed(random_state)

    required_cols = {"CELL", "SOH", "SOC"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must include columns: {required_cols}")

    df_train_parts = []
    selection_log = []

    print("\n===== Auto Split: Selected SOCs for Training =====")

    selected_indices = set()

    # Outer loop: cells
    for cell in train_cells:
        df_cell = df[df["CELL"] == cell]
        if df_cell.empty:
            print(f"Warning: No data for training cell {cell}")
            continue

        # Middle loop: SOHs
        for soh, df_soh in df_cell.groupby("SOH"):
            # Inner loop: SOC ranges
            for (low, high) in train_soc_ranges:
                df_range = df_soh[(df_soh["SOC"] >= low) & (df_soh["SOC"] <= high)]
                if df_range.empty:
                    continue

                unique_socs = sorted(df_range["SOC"].unique())
                chosen_soc = random.choice(unique_socs)

                df_selected = df_soh[df_soh["SOC"] == chosen_soc].copy()
                df_train_parts.append(df_selected)
                selected_indices.update(df_selected.index)

                selection_log.append((cell, soh, (low, high), chosen_soc))
                print(f"Cell={cell}, SOH={soh:.3f}, Range=({low},{high}) → SOC={chosen_soc} "
                      f"({len(df_selected)} samples)")

    # Combine selected training samples
    df_train_full = pd.concat(df_train_parts, ignore_index=False) if df_train_parts else pd.DataFrame(columns=df.columns)

    # Test = all remaining data (unselected SOCs + non-train cells)
    selected_idx_list = list(selected_indices)
    df_test = df.drop(index=selected_idx_list).copy()

    # Split train into train/val
    if len(df_train_full) > 1:
        df_train, df_val = train_test_split(df_train_full, test_size=val_ratio, random_state=random_state, shuffle=True)
    else:
        df_train, df_val = df_train_full.copy(), pd.DataFrame(columns=df.columns)

    # --- Convert to arrays ---
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

    print(f"\nTrain samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Selection log
    selection_df = pd.DataFrame(selection_log, columns=["CELL", "SOH", "SOC_range", "Chosen_SOC"])
    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, selection_df


# For SOC-based splitting with one random SOC per range
def split_by_soc_ranges_random_one_per_range_with_val(
    df: pd.DataFrame,
    train_soc_list: List[Tuple[float, float]],
    features: List[str] = ["R0", "R1", "R2", "R3"],
    target: str = "SOH",
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on SOC ranges.
    For each SOC range in train_soc_list, randomly select ONE SOC value that lies in the range.
    All rows with those selected SOCs go to training (later split 80/20 → train/val),
    and all remaining SOCs in the dataframe go to testing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'SOC' and 'SOH' columns.
    train_soc_list : list of (low, high)
        SOC ranges to consider for potential training SOCs.
    features : list of str
        Feature column names.
    target : str
        Target column name.
    val_ratio : float
        Fraction of training data used as validation.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
    """

    random.seed(random_state)
    np.random.seed(random_state)

    if "SOC" not in df.columns or "SOH" not in df.columns:
        raise ValueError("DataFrame must include 'SOC' and 'SOH' columns.")

    # --- Step 1: Find candidate SOCs in each range ---
    candidate_socs = set()
    for (low, high) in train_soc_list:
        socs_in_range = df.loc[(df["SOC"] >= low) & (df["SOC"] <= high), "SOC"].unique()
        candidate_socs.update(socs_in_range)

    candidate_socs = sorted(candidate_socs)
    if not candidate_socs:
        raise ValueError("No SOCs found in any of the specified ranges.")

    # --- Step 2: Randomly pick one SOC per range ---
    selected_socs = []
    for (low, high) in train_soc_list:
        socs_in_range = [s for s in candidate_socs if low <= s <= high]
        if len(socs_in_range) == 0:
            print(f"⚠️ No SOC found in range ({low}, {high})")
            continue
        chosen_soc = random.choice(socs_in_range)
        selected_socs.append(chosen_soc)
        print(f"Selected training SOC from range ({low}, {high}): {chosen_soc}")

    selected_socs = sorted(set(selected_socs))

    # --- Step 3: Assign train/test based on SOC ---
    df_train_full = df[df["SOC"].isin(selected_socs)].copy()
    df_test = df[~df["SOC"].isin(selected_socs)].copy()

    # --- Step 4: Split train into train/val ---
    if len(df_train_full) > 1:
        df_train, df_val = train_test_split(
            df_train_full, test_size=val_ratio, random_state=random_state, shuffle=True
        )
    else:
        df_train, df_val = df_train_full.copy(), pd.DataFrame(columns=df.columns)

    # --- Step 5: Print group info ---
    def print_group_info(label, dframe):
        if dframe.empty:
            print(f"\n{label}: [empty]")
            return
        grouped = dframe.groupby(["SOH", "SOC"]).size().reset_index(name="count")
        print(f"\n{label} (total {len(dframe)} samples):")
        print(grouped.to_string(index=False))

    print("\n===== SOC-based Split Summary =====")
    print(f"Training SOCs: {selected_socs}")
    print_group_info("Train", df_train)
    print_group_info("Validation", df_val)
    print_group_info("Test", df_test)

    # --- Step 6: Build numpy arrays ---
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

    print("\n===== Summary =====")
    print(f"Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Unique SOCs in training: {selected_socs}")
    print(f"Unique SOCs in test: {sorted(df_test['SOC'].unique())}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test

def split_by_cell_and_rnd_one_soc_per_ranges(
    df: pd.DataFrame,
    train_cells: List[str],
    test_cells: List[str],
    train_soc_ranges: List[Tuple[float, float]],
    features: List[str] = ["R0", "R1", "R2", "R3"],
    target: str = "SOH",
    val_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    Split data by cell and SOC ranges.

    For each train cell and each SOC range:
      - randomly select one SOC value inside the range (if available)
      - use all rows from that SOC value as training
    All data from test_cells go to the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at least ["CELL", "SOC", "SOH"].
    train_cells : list of str
        Cell names to be used for training.
    test_cells : list of str
        Cell names to be used for testing.
    train_soc_ranges : list of (float, float)
        List of SOC ranges for selecting one SOC per range per cell. tuple (a,b) represents range [a,b)
    features : list of str
        Feature column names.
    target : str
        Target column name.
    val_ratio : float
        Fraction of training data used for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
    """
    random.seed(random_state)
    np.random.seed(random_state)

    required_cols = {"CELL", "SOC", "SOH"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must include columns: {required_cols}")

    df_train_list = []
    df_test = df[df["CELL"].isin(test_cells)].copy()

    print("\n===== Per-Cell SOC-based Splitting =====")

    for cell in train_cells:
        df_cell = df[df["CELL"] == cell]
        if df_cell.empty:
            print(f"⚠️ No data found for training cell {cell}")
            continue

        print(f"\nProcessing training cell: {cell}")
        for (low, high) in train_soc_ranges:
            df_range = df_cell[(df_cell["SOC"] >= low) & (df_cell["SOC"] < high)]
            if df_range.empty:
                print(f"  ⚠️ No SOCs found in range ({low}, {high}) for {cell}")
                continue

            unique_socs = sorted(df_range["SOC"].unique())
            chosen_soc = random.choice(unique_socs)

            df_train_part = df_range[df_range["SOC"] == chosen_soc].copy()
            df_train_list.append(df_train_part)

            print(f"  Selected SOC {chosen_soc} for range ({low}, {high})")
            print(f"    -> {len(df_train_part)} samples (SOH, SOC) groups:")
            print(df_train_part.groupby(["SOH", "SOC"]).size().reset_index(name="count").to_string(index=False))

    # Combine all training selections
    df_train_full = pd.concat(df_train_list, ignore_index=True) if df_train_list else pd.DataFrame(columns=df.columns)

    # Split into train / validation
    from sklearn.model_selection import train_test_split
    if len(df_train_full) > 1:
        df_train, df_val = train_test_split(df_train_full, test_size=val_ratio, random_state=random_state, shuffle=True)
    else:
        df_train, df_val = df_train_full.copy(), pd.DataFrame(columns=df.columns)

    # --- Summary printout ---
    def print_group_info(label, dframe):
        if dframe.empty:
            print(f"\n{label}: [empty]")
            return
        grouped = dframe.groupby(["CELL", "SOH", "SOC"]).size().reset_index(name="count")
        print(f"\n{label} (total {len(dframe)} samples):")
        print(grouped.to_string(index=False))

    print("\n===== Summary =====")
    print_group_info("Train", df_train)
    print_group_info("Validation", df_val)
    print_group_info("Test", df_test)

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

    print(f"\nTrain samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test


def split_by_cell_soh_soc_one_selection(
    df: pd.DataFrame,
    train_cells: List[str],
    test_cells: List[str],
    train_soc_ranges: List[Tuple[float, float]],
    features: List[str] = ["R0", "R1", "R2", "R3"],
    target: str = "SOH",
    val_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    For each training cell and each SOH:
      - For each SOC range, randomly select ONE SOC (if any exist within that range)
      - Add all samples for (CELL, SOH, chosen SOC) to training set
    Ignore unselected SOCs from training cells.
    Use ALL data from test_cells as test set.
    Split training data 80/20 into train/validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ["CELL", "SOH", "SOC"] at least.
    train_cells : list of str
        List of cell names to use for training.
    test_cells : list of str
        List of cell names to use for testing (full data used).
    train_soc_ranges : list of (float, float)
        SOC ranges to consider for possible selection.
    features : list of str
        Feature column names.
    target : str
        Target column name.
    val_ratio : float
        Fraction of training data for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
    """
    random.seed(random_state)
    np.random.seed(random_state)

    required_cols = {"CELL", "SOH", "SOC"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must include columns: {required_cols}")

    df_train_parts = []
    selection_log = []

    print("\n===== Multi-Range (CELL, SOH, SOC) Selection =====")

    # Outer loop: cells
    for cell in train_cells:
        df_cell = df[df["CELL"] == cell]
        if df_cell.empty:
            print(f"⚠️ No data for training cell {cell}")
            continue

        # Middle loop: SOHs
        for soh, df_soh in df_cell.groupby("SOH"):
            # Inner loop: SOC ranges
            for (low, high) in train_soc_ranges:
                df_range = df_soh[(df_soh["SOC"] >= low) & (df_soh["SOC"] <= high)]
                if df_range.empty:
                    continue

                unique_socs = sorted(df_range["SOC"].unique())
                chosen_soc = random.choice(unique_socs)
                df_selected = df_soh[df_soh["SOC"] == chosen_soc].copy()
                df_train_parts.append(df_selected)

                selection_log.append((cell, soh, (low, high), chosen_soc))

                print(f"Cell={cell}, SOH={soh:.3f}, Range=({low},{high}) → SOC={chosen_soc} "
                      f"({len(df_selected)} samples)")

    # Combine selected training samples
    df_train_full = pd.concat(df_train_parts, ignore_index=True) if df_train_parts else pd.DataFrame(columns=df.columns)

    # Split into train/val
    if len(df_train_full) > 1:
        df_train, df_val = train_test_split(df_train_full, test_size=val_ratio, random_state=random_state, shuffle=True)
    else:
        df_train, df_val = df_train_full.copy(), pd.DataFrame(columns=df.columns)

    # Test set = all test cells’ data
    df_test = df[df["CELL"].isin(test_cells)].copy()

    # --- Convert to arrays ---
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

    print(f"\nTrain samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Return also the selection log for reproducibility
    selection_df = pd.DataFrame(selection_log, columns=["CELL", "SOH", "SOC_range", "Chosen_SOC"])
    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, selection_df

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
    plt.axis("square")
    savepath = os.path.join(save_dir, f"test{title_CELL}_{title_dataset}.png")
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
    # ---- Load Metadata and Color Map ----
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json"
    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)
    color_map = build_cell_colormap(battery_metadata)

    # ---- Load Data ----
    data_path = Path("fulldf_global_all.csv")
    if data_path is None:
        raise FileNotFoundError("Missing selectdf_global_all.csv.")
    df_all = pd.read_csv(data_path, index_col=0)
    # df_all = df_all[df_all["Temp"] == 25] #NOTE: Select 25 only

    # ---- Clean Data ----
    FEATURES = ["R0", "R1", "R2", "R3", "SOC"] #NOTE: Remove Temp for single Temp case
    # FEATURES = ["R0", "R1", "R2", "R3", "SOC", "Temp"]

    TARGET = "SOH"
    REQ_COLS = ["CELL", "SOH", "SOC", "Temp"] + FEATURES
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
    
    # plot_save_dir = "MLP_plots/allTemp/"
    # if not os.path.exists(plot_save_dir):
    #     os.makedirs(plot_save_dir)

    ####### By CELL Split DATA ########
    # train_cells = ["CELL009", "CELL021", "CELL077"] + ["CELL032", "CELL070", "CELL101"] # all 0 & 45
    # test_cells = ["CELL013", "CELL042", "CELL045", "CELL050", "CELL054", "CELL076", "CELL090", "CELL096"] # all 25
    # train_cells = [ "CELL013","CELL045","CELL042","CELL054","CELL076","CELL090", "CELL096"]
    # test_cells = ["CELL050",]
    # train_cells = ["CELL009","CELL021", ]
    # test_cells = ["CELL077"]
    train_cells = ["CELL032", "CELL070"]
    test_cells = ["CELL101"]


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
    plot_save_dir = "MLP_plots/leave_one_out_test45/"
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)


    ####### By SOC range Split DATA ########
    # train_cells = ["CELL013", "CELL042", "CELL045", "CELL050", "CELL054", "CELL076","CELL090","CELL096"] # all 25 
    # train_soc_ranges = [(0.25,0.50), (0.50, 0.80), (0.80,1)]

    # X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, selection_df = split_by_soc_one_selection_else_to_test(
    #     df=df_all,
    #     train_cells=train_cells,
    #     train_soc_ranges=train_soc_ranges,
    #     features=FEATURES,
    #     target="SOH",
    #     val_ratio=0.2,
    #     random_state=42,
    # )
    # # print("\n=== Selection Log ===")
    # # print(selection_df)

    # plot_save_dir = "MLP_plots/Split_bySOC/"
    # if not os.path.exists(plot_save_dir):
    #     os.makedirs(plot_save_dir)

    ####### By CELL and SOC ranges (random one) Split DATA ########
    # train_cells = ["CELL042", "CELL050", "CELL054", "CELL076", "CELL090", "CELL096"] # all 25 (exclude CELL045)
    # train_cells = [ "CELL013", "CELL042", "CELL045", "CELL050","CELL054",  "CELL076","CELL090",] # all 25 

    # test_cells = ["CELL096"]
    # train_soc_ranges = [(0.25,0.50), (0.50, 0.80), (0.80,1)]

    # 1. Random select one soc across all SOH
    # X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = \
    #     split_by_cell_and_rnd_one_soc_per_ranges(
    #         df=df_all,
    #         train_cells=train_cells,
    #         test_cells=test_cells,
    #         train_soc_ranges=train_soc_ranges,
    #         features=FEATURES,
    #         target=TARGET,
    #         val_ratio=0.2,
    #         random_state=42
    #     )
    
    # 2. Random select one soc per SOH
    # X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, selection_df = \
    # split_by_cell_soh_soc_one_selection(
    #     df=df_all,
    #     train_cells=train_cells,
    #     test_cells=test_cells,
    #     train_soc_ranges=train_soc_ranges,
    #     features=FEATURES,
    #     target=TARGET,
    #     val_ratio=0.2,
    #     random_state=42,
    # )
    # plot_save_dir = "MLP_plots/leave_one_out_test25/select_SOC"
    # if not os.path.exists(plot_save_dir):
    #     os.makedirs(plot_save_dir)

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
        save_path="MLP_predictions_trainset.csv",
        plot_save_dir=plot_save_dir,
        plot_title_CELL=test_cells[0],
        plot_title_dataset="train"
    )



    #TODO: CHANGE PATHs
    # ---- Validation Visualization ----
    val_df_vis, val_mae, val_rmse, val_mape = evaluate_and_plot(
        model,
        X_val,
        y_val,
        df_val,
        "SOH Prediction on Validation Data",
        color_map,
        save_path="MLP_predictions_valset.csv",
        plot_save_dir=plot_save_dir,
        plot_title_CELL=test_cells[0],
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
        save_path="MLP_predictions_testset.csv",
        plot_save_dir=plot_save_dir,
        plot_title_CELL=test_cells[0],
        plot_title_dataset="test"
    )


if __name__ == "__main__":
    main()
