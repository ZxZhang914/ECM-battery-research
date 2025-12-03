import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import json
import os


SEED = 42
np.random.seed(SEED)


# =====================================================
# Define helper functions
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

def zscore_with_stats(series, mean, std):
    return (series - mean) / std


def evaluate(y_true, y_pred, label="Set"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {label} Performance ===")
    print(f"R²   = {r2:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")
    return r2, mae, rmse, mape

def eval_model(df_split, name, aggregate=False):
    if aggregate:
        mask = df_split["SOH_mean"].notna() & df_split["Pred_mean"].notna()
        y_true = df_split.loc[mask, "SOH_mean"]
        y_pred = df_split.loc[mask, "Pred_mean"]
    else:
        mask = df_split["SOH"].notna() & df_split["_pred_OLS"].notna()
        y_true = df_split.loc[mask, "SOH"]
        y_pred = df_split.loc[mask, "_pred_OLS"]

    if mask.any():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)

        print(f"\n=== {name} Performance ===")
        print(f"R²   = {r2:.4f}")
        print(f"MAE  = {mae:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAPE = {mape:.2f}%")
    else:
        print(f"\nNo valid data to evaluate for {name}.")
    
    return r2, mae, rmse, mape

def plot_pred_vs_actual_loo(df_train, df_test, test_cell, aggregate=False, color_map=None, save_dir="MLP_plots/LOOCV25"):
    """
    Plot predicted vs actual SOH for Leave-One-Cell-Out setup:
    - Training cells plotted in gray
    - Test cell plotted in color (from color_map)
    - Handles both raw and aggregated (mean-based) data
    """
    plt.figure(figsize=(6, 6))
    plot_range = [1.45, 3.75]

    if color_map is None:
        unique_cells = df_test["CELL"].unique()
        cmap = plt.cm.get_cmap("tab10", len(unique_cells))
        color_map = {cell: cmap(i) for i, cell in enumerate(unique_cells)}

   
    if aggregate:
        train_mask = df_train["SOH_mean"].notna() & df_train["Pred_mean"].notna()
        test_mask  = df_test["SOH_mean"].notna() & df_test["Pred_mean"].notna()
        train_x, train_y = df_train.loc[train_mask, "SOH_mean"], df_train.loc[train_mask, "Pred_mean"]
        test_x,  test_y  = df_test.loc[test_mask, "SOH_mean"],  df_test.loc[test_mask, "Pred_mean"]
    else:
        train_mask = df_train["SOH"].notna() & df_train["_pred_OLS"].notna()
        test_mask  = df_test["SOH"].notna() & df_test["_pred_OLS"].notna()
        train_x, train_y = df_train.loc[train_mask, "SOH"], df_train.loc[train_mask, "_pred_OLS"]
        test_x,  test_y  = df_test.loc[test_mask, "SOH"],  df_test.loc[test_mask, "_pred_OLS"]

   
    plt.scatter(train_x, train_y, alpha=0.5, s=20, color="gray", label="Training")
    plt.scatter(test_x, test_y, alpha=0.9, s=20, color=color_map[test_cell], label=f"Test ({test_cell})")
    plt.plot(plot_range, plot_range, "r--", label="y = x")

    # Labels and aesthetics
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(f"OLS Leave-One-Out — Test CELL {test_cell}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis("square")
    plt.xlim(plot_range[0], plot_range[1])
    plt.ylim(plot_range[0], plot_range[1])
    plt.tight_layout()
    plot_name = "combined_val_test_agg.png" if aggregate else "combined_val_test.png"
    plt.savefig(os.path.join(save_dir, plot_name), dpi=300)
    plt.close()


def compute_group_performance(df_split):
    required_cols = ["CELL", "SOH", "SOC", "_pred_OLS"]
    required_cols = ["CELL", "SOH",  "_pred_OLS"] # NOTE

    if not all(c in df_split.columns for c in required_cols):
        raise ValueError(f"Missing required columns in df_split: {required_cols}")

    group_results = []
    for (cell, soh, soc), sub in df_split.groupby(["CELL", "SOH", "SOC"]):
        if len(sub) < 2:
            continue
        sub_sorted = sub.sort_values("_pred_OLS").reset_index(drop=True)
        lower = np.percentile(sub_sorted["_pred_OLS"], 25)
        upper = np.percentile(sub_sorted["_pred_OLS"], 75)
        mid = sub_sorted[(sub_sorted["_pred_OLS"] >= lower) & (sub_sorted["_pred_OLS"] <= upper)]
        if mid.empty:
            mid = sub_sorted.iloc[[len(sub_sorted)//2]] if len(sub_sorted) >= 1 else sub_sorted
        group_results.append({
            "CELL": cell,
            "SOH": soh,
            "SOC": soc,
            "SOH_mean": mid["SOH"].mean(),
            "Pred_mean": mid["_pred_OLS"].mean(),
            "Pred_std": mid["_pred_OLS"].std(),
            "N_mid": len(mid)
        })
    return pd.DataFrame(group_results)

# =====================================================
# Helper for Experiment 3 – Reduced SOC sampling
# =====================================================
def reduce_training_samples(df_train, cell_col="CELL", soc_col="SOC", soh_col="SOH", random_state=42):
    """
    For each CELL and each SOH, select 3 SOC points (one per range)
    and include ALL rows corresponding to those SOCs (e.g., all trials).
    """
    np.random.seed(random_state)
    soc_ranges = [(0.25, 0.5), (0.5, 0.8), (0.8, 1.0)]
    selected_soc_values = []

    for cell, cell_group in df_train.groupby(cell_col):
        for soh, soh_group in cell_group.groupby(soh_col):
            for (low, high) in soc_ranges:
                subset = soh_group[(soh_group[soc_col] >= low) & (soh_group[soc_col] < high)]
                if not subset.empty:
                    # randomly select one SOC value
                    soc_value = subset[soc_col].sample(1, random_state=random_state).iloc[0]
                    selected_soc_values.append((cell, soh, soc_value))

    # Include all rows for those SOC values (i.e., all trials)
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


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Linear Regression Leave-One-Cell-Out Cross-Validation")
    
    parser.add_argument("--partial_SOC", action="store_true", help="Use reduced training samples (Experiment 3 condition)")
    parser.add_argument("--input", "-i", type=str, default="fulldf_removeAbOod_date_G25SOC_all.csv", help="Input CSV data file")
    parser.add_argument("--output", "-o", type=str, default="MLP_plots/LOOCV25", help="Output directory for plots and results")
    parser.add_argument("--features", "-f", type=str, nargs="+", default=["R0", "R1", "R2", "R3", "SOC"], help="List of feature columns to use")
    
    args = parser.parse_args()
   
    REDUCED_TRAINING = args.partial_SOC
    input_file = args.input
    output_dir = args.output
    FEATURES = args.features

    print("Using features:", FEATURES)
    if REDUCED_TRAINING:
        print("Using Partial SOC for training.")


    # Load data
    df_all = pd.read_csv(input_file, index_col=0)
    df_all = df_all[df_all["Temp"] == 25]  # Use only 25°C data

    cells = sorted(df_all["CELL"].unique())  # or custom cell list

    # Color map for consistent plotting
    # cmap = mpl.colormaps.get_cmap("tab10").resampled(len(cells))
    # COLOR_MAP = {cell: cmap(i) for i, cell in enumerate(cells)}
     # Build color map
    with open("../EVC_EIS_Data/original_data/Battery_Info_DRT.json") as f:
        metadata = json.load(f)
    COLOR_MAP = build_cell_colormap(metadata)


    # To store results
    results = []

    # --- Leave-One-Cell-Out Loop ---
    for test_cell in cells:
        print("\n" + "=" * 60)
        print(f"Leave-One-Out: Testing on {test_cell}")

        train_df = df_all[df_all["CELL"] != test_cell].copy()
        # =====================================================
        # (Optional) Apply SOC reduction — Experiment 3 condition
        # =====================================================
        # REDUCED_TRAINING = False  # NOTE: toggle ON for Experiment 3
        if REDUCED_TRAINING:
            train_df = reduce_training_samples(train_df)
        test_df = df_all[df_all["CELL"] == test_cell].copy()
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Compute training means/stds
        train_means = {col: train_df[col].mean() for col in FEATURES }
        train_stds  = {col: train_df[col].std(ddof=0) for col in FEATURES}

        # Standardize both train/test using training stats
        for col in FEATURES: #NOTE: standardized features
            train_df[f"{col}_z"] = zscore_with_stats(train_df[col], train_means[col], train_stds[col])
            test_df[f"{col}_z"]  = zscore_with_stats(test_df[col], train_means[col], train_stds[col])

        # Fit OLS model
        rhs_terms = " + ".join(f"{col}_z" for col in FEATURES)
        formula = f"SOH ~ {rhs_terms}"
        print("OLS formula:", formula)

        ols = smf.ols(formula, data=train_df).fit()
        print(ols.summary())

        # Predict
        train_df["_pred_OLS"] = ols.predict(train_df)
        test_df["_pred_OLS"]  = ols.predict(test_df)

        # Evaluate
        train_metrics = evaluate(train_df["SOH"], train_df["_pred_OLS"], "Training")
        test_metrics  = evaluate(test_df["SOH"], test_df["_pred_OLS"], f"Testing (CELL={test_cell})")
        exp_tag = "_Exp3_reducedSOC" if REDUCED_TRAINING else ""
        plot_dir = f"{output_dir}/{exp_tag}/{test_cell}/" #NOTE: change save dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot raw
        plot_pred_vs_actual_loo(train_df, test_df, test_cell, aggregate=False, color_map=COLOR_MAP, save_dir=plot_dir)
        
        # aggregated performance
        train_perf_df = compute_group_performance(train_df)
        test_perf_df  = compute_group_performance(test_df)
        # Save
        train_perf_df.to_csv(f"{plot_dir}/LR_predictions_valset_aggregated.csv", index=False)
        test_perf_df.to_csv(f"{plot_dir}/LR_predictions_testset_aggregated.csv", index=False)
        
        
        train_agg_metrics = eval_model(train_perf_df, "Aggregate Training", aggregate=True)
        test_agg_metrics = eval_model(test_perf_df, f"Aggregate Testing (CELL={test_cell})", aggregate=True)
        
        # Plot aggregated
        plot_pred_vs_actual_loo(train_perf_df, test_perf_df, test_cell, aggregate=True, color_map=COLOR_MAP, save_dir=plot_dir)
        
        # Store metrics
        results.append({
            "Cell": test_cell,
            "R2_train": train_metrics[0],
            "MAE_train": train_metrics[1],
            "RMSE_train": train_metrics[2],
            "MAPE_train": train_metrics[3],
            "R2_test": test_metrics[0],
            "MAE_test": test_metrics[1],
            "RMSE_test": test_metrics[2],
            "MAPE_test": test_metrics[3],
            "R2_train_agg": train_agg_metrics[0],
            "MAE_train_agg": train_agg_metrics[1],
            "RMSE_train_agg": train_agg_metrics[2],
            "MAPE_train_agg": train_agg_metrics[3],
            "R2_test_agg": test_agg_metrics[0],
            "MAE_test_agg": test_agg_metrics[1],
            "RMSE_test_agg": test_agg_metrics[2],
            "MAPE_test_agg": test_agg_metrics[3],
        })


    # Save Result
    results_df = pd.DataFrame(results)
    exp_tag = "_Exp3_reducedSOC" if REDUCED_TRAINING else ""
    results_df.to_csv(f"{output_dir}/ols_leave_one_cell_out_metrics{exp_tag}.csv", index=False)
    print(f"\nSaved metrics to {output_dir}/ols_leave_one_cell_out_metrics{exp_tag}.csv")

    print("\n=== Overall Summary ===")
    print(results_df[["Cell", "R2_test", "MAE_test", "RMSE_test", "MAPE_test", "R2_test_agg", "MAE_test_agg", "RMSE_test_agg", "MAPE_test_agg"]])


if __name__ == "__main__":
    main()
