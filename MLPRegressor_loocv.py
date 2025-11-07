import os
import json
import math
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

from MLPModel import *

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

def split_leave_one_out(df, leave_out_cell, features, target="SOH", val_ratio=0.2, random_state=42):
    """
    Perform Leave-One-Cell-Out split.
    - leave_out_cell: the cell used as the test set
    - remaining cells are split 80/20 into train/validation
    """
    # Separate test cell
    df_test = df[df["CELL"] == leave_out_cell].copy()
    df_trainval = df[df["CELL"] != leave_out_cell].copy()

    # Split remaining cells into train/validation
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_ratio, random_state=random_state, shuffle=True
    )

    # Convert to arrays
    X_train = df_train[features].values.astype(np.float32)
    y_train = df_train[target].values.astype(np.float32)
    X_val = df_val[features].values.astype(np.float32)
    y_val = df_val[target].values.astype(np.float32)
    X_test = df_test[features].values.astype(np.float32)
    y_test = df_test[target].values.astype(np.float32)

    print(f"Leave-one-out test cell: {leave_out_cell}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test


def plot_val_test_together(df_val, df_test, color_map, save_dir, tag, title="Validation vs Test Comparison"):
    """
    Plot validation (gray) and test (colored by cell) predictions together.
    Assumes df_val and df_test each contain 'y_true', 'y_pred', and 'CELL' columns.
    """
    plt.figure(figsize=(6, 6))
    plot_range = [1.45, 3.75]

    # Validation points (gray)
    plt.scatter(df_val["y_true"], df_val["y_pred"], color="gray", s=10, alpha=0.6, label="Validation")

    # Test points (colored by cell)
    plt.scatter(df_test["y_true"], df_test["y_pred"],
                c=df_test["CELL"].map(color_map), s=10, alpha=0.85, label="Test")

    plt.plot(plot_range, plot_range, "k--", lw=2, label="Ideal = y=x")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(title)
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.axis("square")
    plt.grid(True, alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color="gray", label="Validation")]
    handles += [
        mpatches.Patch(color=color_map[cell], label=cell)
        for cell in sorted(df_test["CELL"].unique()) if cell in color_map
    ]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_val_test.png"), dpi=300)
    plt.close()


def evaluate_and_plot(model, X, y, df_subset, title, color_map, save_dir, tag):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X).float().to(DEVICE)).squeeze(1).cpu().numpy()

    mae = mean_absolute_error(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    mape = np.mean(np.abs((y - y_pred) / np.clip(np.abs(y), 1e-6, None))) * 100
    r2 = r2_score(y, y_pred)

    print(f"{title}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")

    # Attach predictions for later use
    df_vis = df_subset.copy()
    df_vis["y_true"] = y
    df_vis["y_pred"] = y_pred

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, c=df_subset["CELL"].map(color_map), s=20, alpha=0.8)
    plt.plot([1.45, 3.75], [1.45, 3.75], "k--", lw=2, label="Ideal = y=x")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(title)
    plt.axis("square")
    plt.grid(True, alpha=0.3)
    plt.xlim(1.45, 3.75)
    plt.ylim(1.45, 3.75)

    handles = [mpatches.Patch(color=color_map[cell], label=cell)
               for cell in sorted(df_subset["CELL"].unique()) if cell in color_map]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}.png"), dpi=300)
    plt.close()

    return df_vis, mae, rmse, mape, r2

# =====================================================
# Additional helper for Experiment 3 (Reduced Training)
# =====================================================
def reduce_training_samples(df_train, cell_col="CELL", soc_col="SOC", soh_col="SOH", random_state=42):
    """
    For each CELL and each SOH, select 3 SOC points (one per range)
    and include ALL rows corresponding to those SOCs (e.g., all trials).

    SOC ranges:
        [0.25–0.5)
        [0.5–0.8)
        [0.8–1.0]
    """
    np.random.seed(random_state)
    soc_ranges = [(0.25, 0.5), (0.5, 0.8), (0.8, 1.0)]
    selected_soc_values = []

    for cell, cell_group in df_train.groupby(cell_col):
        for soh, soh_group in cell_group.groupby(soh_col):
            for (low, high) in soc_ranges:
                subset = soh_group[(soh_group[soc_col] >= low) & (soh_group[soc_col] < high)]
                if not subset.empty:
                    # Randomly select one SOC value, but not just one row
                    soc_value = subset[soc_col].sample(1, random_state=random_state).iloc[0]
                    selected_soc_values.append((cell, soh, soc_value))

    # Now include all rows corresponding to these selected SOCs (i.e., all trials)
    selected_rows = []
    for (cell, soh, soc_value) in selected_soc_values:
        mask = (
            (df_train[cell_col] == cell)
            & (df_train[soh_col] == soh)
            & (np.isclose(df_train[soc_col], soc_value, atol=1e-6))
        )
        selected_rows.append(df_train[mask])

    reduced_df = pd.concat(selected_rows, ignore_index=True)

    print(f"Reduced training samples: {len(reduced_df)} "
          f"({len(reduced_df) / len(df_train) * 100:.1f}% of original)")
    return reduced_df

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
    return mae, rmse, mape, r2


def plot_aggregated_val_test(df_perf_val, df_perf_test, color_map=None, save_dir="MLP_plots/LOOCV25/features_allRC"):
    plt.figure(figsize=(6, 6))
    plot_range = [1.45, 3.75]

    # Validation points (gray)
    plt.scatter(df_perf_val["SOH_mean"], df_perf_val["Pred_mean"], color="gray", s=10, alpha=0.6, label="Validation")

    # Test points (colored by cell)
    plt.scatter(df_perf_test["SOH_mean"], df_perf_test["Pred_mean"],
                c=df_perf_test["CELL"].map(color_map), s=10, alpha=0.85, label="Test")

    plt.plot(plot_range, plot_range, "k--", lw=2, label="Ideal = y=x")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Validation vs Test Comparison")
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.axis("square")
    plt.grid(True, alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color="gray", label="Validation")]
    handles += [
        mpatches.Patch(color=color_map[cell], label=cell)
        for cell in sorted(df_perf_test["CELL"].unique()) if cell in color_map
    ]
    plt.legend(handles=handles, title="Cell", bbox_to_anchor=(1.05, 1), loc="upper left")

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"combined_val_test_agg.png"), dpi=300)
    plt.close()

# =====================================================
# Main Leave-One-Cell-Out
# =====================================================
def main():
    # Load data
    df_all = pd.read_csv("fulldf_global_all.csv", index_col=0)
    df_all = df_all[df_all["Temp"] == 25]  # Use only 25°C data
    FEATURES = ["R0", "R1", "R2", "R3", "SOC"]
    TARGET = "SOH"

    # Clean data
    df_all = df_all.dropna(subset=["CELL", "SOH", "SOC", "Temp"] + FEATURES)
    cells = sorted(df_all["CELL"].unique())
    print(f"Detected {len(cells)} cells: {cells}")

    # Build color map
    with open("../EVC_EIS_Data/original_data/Battery_Info_DRT.json") as f:
        metadata = json.load(f)
    color_map = build_cell_colormap(metadata)

    results = []
    agg_results = []

    # --- Leave-One-Cell-Out Loop ---
    for test_cell in cells:
        print("\n" + "=" * 60)
        print(f"🔹 Leave-One-Out: Testing on {test_cell}")

        # Split normally
        X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = \
            split_leave_one_out(df_all, leave_out_cell=test_cell,
                                features=FEATURES, target=TARGET)

        # =====================================================
        # 🔹 Experiment 3 condition: reduce training SOC samples
        # =====================================================
        REDUCED_TRAINING = False  # NOTE: toggle here for Experiment 3
        if REDUCED_TRAINING:
            df_train = reduce_training_samples(df_train, soc_col="SOC", soh_col="SOH")
            # Rebuild numpy arrays
            X_train = df_train[FEATURES].values.astype(np.float32)
            y_train = df_train[TARGET].values.astype(np.float32)

        # Standardize using training stats only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # DataLoaders
        train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=256, shuffle=False)

        # Model
        model = MLPRegressor(in_dim=len(FEATURES), hidden=[128, 64, 32], p_drop=0.1).to(DEVICE)

        # Train
        model = train(model, train_loader, val_loader, epochs=400)

        # Evaluate
        exp_tag = "_Exp3_reducedSOC" if REDUCED_TRAINING else ""
        plot_dir = f"MLP_plots/LOOCV25{exp_tag}/Rs_SOC/{test_cell}/"
        os.makedirs(plot_dir, exist_ok=True)

        train_df_vis, train_mae, train_rmse, train_mape, train_r2 = evaluate_and_plot(
            model, X_train, y_train, df_train, f"Train — Leave-Out {test_cell}", 
            color_map, plot_dir, "train"
        )

        val_df_vis, val_mae, val_rmse, val_mape, val_r2 = evaluate_and_plot(
            model, X_val, y_val, df_val, f"Validation — Leave-Out {test_cell}", 
            color_map, plot_dir, "val"
        )

        test_df_vis, test_mae, test_rmse, test_mape, test_r2 = evaluate_and_plot(
            model, X_test, y_test, df_test, f"Test — Leave-Out {test_cell}", 
            color_map, plot_dir, "test"
        )

        # Combined val+test visualization
        plot_val_test_together(val_df_vis, test_df_vis, color_map, plot_dir,
                            tag="combined", title=f"Validation vs Test — Leave-Out {test_cell}")

        results.append({
            "test_cell": test_cell,
            "train_R2": train_r2, "val_R2": val_r2, "test_R2": test_r2,
            "train_MAE": train_mae, "val_MAE": val_mae, "test_MAE": test_mae,
            "train_RMSE": train_rmse, "val_RMSE": val_rmse, "test_RMSE": test_rmse,
            "train_MAPE": train_mape, "val_MAPE": val_mape, "test_MAPE": test_mape,
        })

         # ================================================================
        # Compute and visualize aggregated results
        # ================================================================
        df_val_performance = compute_group_performance(val_df_vis)
        df_test_performance = compute_group_performance(test_df_vis)

        # Save
        df_val_performance.to_csv(f"{plot_dir}/MLP_predictions_valset_aggregated.csv", index=False)
        df_test_performance.to_csv(f"{plot_dir}/MLP_predictions_testset_aggregated.csv", index=False)

        # Evaluate aggregated performance
        val_agg_mae, val_agg_rmse, val_agg_mape, val_agg_r2 = eval_model(df_val_performance, "Validation (Aggregated)", aggregate=True)
        test_agg_mae, test_agg_rmse, test_agg_mape, test_agg_r2 = eval_model(df_test_performance, "Test (Aggregated)", aggregate=True)

        # Plot aggregated performance
        plot_aggregated_val_test(df_val_performance, df_test_performance, color_map, save_dir=plot_dir)

        agg_results.append({
            "test_cell": test_cell,
            "val_agg_R2": val_agg_r2, "test_agg_R2": test_agg_r2,
            "val_agg_MAE": val_agg_mae, "test_agg_MAE": test_agg_mae,
            "val_agg_RMSE": val_agg_rmse, "test_agg_RMSE": test_agg_rmse,
            "val_agg_MAPE": val_agg_mape, "test_agg_MAPE": test_agg_mape,
        })


    # Save summary
    results_df = pd.DataFrame(results)
    # results_df.to_csv(f"MLP_LOOCV_results{exp_tag}.csv", index=False)
    # print(f"\nSaved all LOOCV results → MLP_LOOCV_results_{exp_tag}.csv")
    print("\n=== Leave-One-Out Summary ===")
    print(results_df[["test_cell", "test_R2", "test_MAE", "test_RMSE", "test_MAPE"]].round(4))

    agg_results_df = pd.DataFrame(agg_results)
    print("\n=== Leave-One-Out Agg Summary ===")
    print(agg_results_df[["test_cell", "test_agg_R2", "test_agg_MAE", "test_agg_RMSE", "test_agg_MAPE"]].round(4))



if __name__ == "__main__":
    main()
