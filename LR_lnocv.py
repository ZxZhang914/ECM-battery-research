import os
import json
import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import math
import sys
sys.stdout.reconfigure(encoding='utf-8')
# =====================================================
# Config
# =====================================================
SEED = 42
np.random.seed(SEED)


# =====================================================
# Helper Functions
# =====================================================
def build_cell_colormap(metadata, shade_min=0.30, shade_max=0.90):
    """Same as before — build color map by temperature group."""
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
    """Reduced training condition (optional)."""
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


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


# =====================================================
# Leave-N-Out for OLS
# =====================================================
def leave_n_out_OLS(df_all, n, reduced_training=True):
    """
    Generalized Leave-N-Out cross-validation using OLS regression.
    """
    all_cells = sorted(df_all["CELL"].unique())
    combs = list(itertools.combinations(all_cells, n))
    print(f"\n* Running Leave-{n}-Out ({len(combs)} combinations)")
    results = []

    for test_cells in combs:
        test_cells = list(test_cells)
        print(f"\n{'='*60}\n** Leave-{n}-Out Test Cells: {test_cells}")

        # Split data
        df_test = df_all[df_all["CELL"].isin(test_cells)].copy()
        df_trainval = df_all[~df_all["CELL"].isin(test_cells)].copy()
        df_train, df_val = train_test_split(df_trainval, test_size=0.2, random_state=SEED, shuffle=True)

        if reduced_training:
            df_train = reduce_training_samples(df_train)

        # Z-score normalization (fit on training only)
        scaler = StandardScaler()
        z_features = ["R0", "R1", "R2", "R3", "SOC"]

        # Fit only on training data
        scaler.fit(df_train[z_features])

        for df_ in [df_train, df_val, df_test]:
            scaled = scaler.transform(df_[z_features])
            for i, f in enumerate(z_features):
                df_[f"{f}_z"] = scaled[:, i]

        # Fit OLS
        formula = "SOH ~ R0_z + R1_z + R2_z + R3_z + SOC_z"
        ols = smf.ols(formula, data=df_train).fit()

        # Predict
        df_train["_pred_OLS"] = ols.predict(df_train)
        df_val["_pred_OLS"] = ols.predict(df_val)
        df_test["_pred_OLS"] = ols.predict(df_test)

        # Evaluate full test
        test_mae, test_rmse, test_mape, test_r2 = evaluate_regression(df_test["SOH"], df_test["_pred_OLS"])
        print(f"Test — MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%, R2={test_r2:.4f}")

        # Per-cell results
        for cell in test_cells:
            df_cell = df_test[df_test["CELL"] == cell]
            if len(df_cell) > 0:
                mae, rmse, mape, r2 = evaluate_regression(df_cell["SOH"], df_cell["_pred_OLS"])
                results.append({
                    "n": n,
                    "test_cell": cell,
                    "combo": "_".join(test_cells),
                    "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2
                })

    return pd.DataFrame(results)


# =====================================================
# Plotting
# =====================================================
def plot_leave_n_out_results(results_df, save_dir="OLS_plots/LeaveNOut"):
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
    df_all = df_all[df_all["Temp"] == 25]
    df_all = df_all.dropna(subset=["CELL", "SOH", "SOC", "Temp", "R0", "R1", "R2", "R3"])
    print(f"Detected {df_all['CELL'].nunique()} cells.")

    # Load metadata for colors (optional)
    with open("../EVC_EIS_Data/original_data/Battery_Info_DRT.json") as f:
        metadata = json.load(f)
    color_map = build_cell_colormap(metadata)

    # Save dir
    save_dir = f"OLS_plots/LeaveNOut"
    os.makedirs(save_dir, exist_ok=True)

    # Run for n = 3, 2, 1
    N_values = [7,6,5,4,3,2,1]
    all_results = []
    for n in N_values:
        df_n = leave_n_out_OLS(df_all, n, reduced_training=True)
        all_results.append(df_n)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(f"{save_dir}/OLS_LOON_results_all.csv", index=False)
    print("\nSaved: OLS_LOON_results_all.csv")

    plot_leave_n_out_results(results_df, save_dir)
    print("Saved plot: LeaveNOut_OLS_MAPE_summary.png")


if __name__ == "__main__":
    main()
