import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import json
import argparse

from utils import format_EIS, EIS_Nyquist_meas_vs_fit_save


def collect_rmse_rel_for_models(
    base_dir,
    cell,
    soh_soc_map,
    ecm_models
):
    if not isinstance(ecm_models, (list, tuple)) or len(ecm_models) == 0:
        raise ValueError("ecm_models must be a non-empty list of model names")

    records = []

    for soh, n_soc in soh_soc_map.items():
        for soc in range(1, n_soc + 1):
            label = f"SOH{soh}_SOC{soc}"

            for model in ecm_models:
                file_path = os.path.join(
                    base_dir,
                    f"{cell}",
                    f"{model}_RMSE_trials100",
                    f"soh{soh}",
                    f"{cell}_soh{soh}_soc{soc}_trials100_objFunc_RMSE_{model}_rmOutliers2.csv"
                )

                if not os.path.exists(file_path):
                    print(f"Missing: {file_path}")
                    continue

                df = pd.read_csv(file_path)
                if df.shape[0] == 0:
                    print(f"Empty CSV: {file_path}")
                    continue

                best = df.iloc[0]

                if "RMSE_rel" not in best.index:
                    print(f"Missing RMSE_rel column: {file_path}")
                    continue

                records.append({
                    "Label": label,
                    "Model": model,
                    "RMSE_rel": best["RMSE_rel"]
                })

    if len(records) == 0:
        raise ValueError("No records collected. Check file paths or SOH_SOC map.")

    return pd.DataFrame(records)


def plot_rmse_rel_comparison(cell, df, models=None):
    plt.figure(figsize=(1.4 * df["Label"].nunique() + 3, 5))
    sns.barplot(
        data=df,
        x="Label",
        y="RMSE_rel",
        hue="Model"
    )
    plt.xlabel("(SOH, SOC)")
    plt.ylabel("RMSE_rel")

    if models is None:
        title_models = ", ".join(df["Model"].unique())
    else:
        title_models = ", ".join(models)

    plt.title(f"{cell} Best Trial's RMSE_rel Comparison: {title_models}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def load_trials_df(base_dir, cell, model, soh, soc):
    file_path = os.path.join(
        base_dir,
        f"{cell}",
        f"{model}_RMSE_trials100",
        f"soh{soh}",
        f"{cell}_soh{soh}_soc{soc}_trials100_objFunc_RMSE_{model}_rmOutliers2.csv"
    )

    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        return None

    df = pd.read_csv(file_path)
    if df.shape[0] == 0:
        print(f"Empty CSV: {file_path}")
        return None

    return df


def build_param_long_for_soh_soc(base_dir, cell, soh, soc, models, exclude_cols):
    parts = []

    for model in models:
        df = load_trials_df(base_dir, cell, model, soh, soc)
        if df is None:
            continue

        param_cols = [c for c in df.columns if c not in exclude_cols]
        if len(param_cols) == 0:
            continue

        # keep only numeric columns (some may be strings like JSON)
        df_params = df[param_cols].apply(pd.to_numeric, errors="coerce")

        # drop columns that are fully NaN after coercion
        df_params = df_params.dropna(axis=1, how="all")
        if df_params.shape[1] == 0:
            continue

        df_long = df_params.melt(var_name="Parameter", value_name="Value")
        df_long["Model"] = model
        df_long = df_long.dropna(subset=["Value"])

        parts.append(df_long)

    if len(parts) == 0:
        return None

    out = pd.concat(parts, ignore_index=True)
    out["SOH"] = soh
    out["SOC"] = soc
    out["Label"] = f"SOH{soh}_SOC{soc}"
    return out


def plot_param_density_grid_for_label(df_long, cell, label, ncols=4, bw_adjust=1.0):
    if df_long is None or df_long.shape[0] == 0:
        return

    params = sorted(df_long["Parameter"].unique())
    n_params = len(params)
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, p in enumerate(params):
        ax = axes[i]
        sub = df_long[df_long["Parameter"] == p].copy()
        if sub.shape[0] == 0:
            ax.axis("off")
            continue

        # If too few points overall, KDE can be unreliable; still try
        try:
            sns.kdeplot(
                data=sub,
                x="Value",
                hue="Model",
                ax=ax,
                common_norm=False,
                bw_adjust=bw_adjust,
                fill=False
            )
        except Exception:
            sns.histplot(
                data=sub,
                x="Value",
                hue="Model",
                ax=ax,
                stat="density",
                element="step",
                common_norm=False
            )

        ax.set_title(p)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    # turn off unused axes
    for j in range(n_params, len(axes)):
        axes[j].axis("off")


    fig.suptitle(f"{cell} Parameter Densities (All Trials) — {label}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    sns.set(style="whitegrid")

    # Configuration
    parser = argparse.ArgumentParser(description="ECM result configuration")
    parser.add_argument("--cell-name", type=str, default="CELL042", help="Cell name identifier")
    parser.add_argument("--ecm-name", type=str, default="v3CM9", help="ECM model name")
    parser.add_argument("--obj-func", type=str, default="RMSE", choices=["RMSE", "MAE", "MSE"], help="Objective function")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--ecm-list", type=str, nargs='+', default=["v3CM9", "v3CM10"], help="List of ECM model names to compare")
    args = parser.parse_args()
    
    CELL_NAME = args.cell_name
    ECM_name = args.ecm_name
    obj_func = args.obj_func
    num_trials = args.num_trials
    ECM_list = args.ecm_list
    BASE_DIR = "ECM_Params_Estimation"



    # ==== Load battery metadata from JSON file ====
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json" # Check the path
    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)   # <--- this is now a dict
    celli_metadata = battery_metadata[CELL_NAME]
    print(f"Temperature: {celli_metadata["temperature"]}")
    n_soh = celli_metadata["num_soh"]

    SOH_SOC_MAP = {}
    for i in range(n_soh):
        SOH_SOC_MAP[i+1] =  celli_metadata["soh"][i]["num_soc"]
    
    print(f"{CELL_NAME} SOH_SOC_MAP: {SOH_SOC_MAP}")

    # --- Compare different ECM structures' RMSE_rel ------
    df_rmse_rel = collect_rmse_rel_for_models(
        base_dir="ECM_Params_Estimation",
        cell=CELL_NAME,
        soh_soc_map=SOH_SOC_MAP,
        ecm_models=ECM_list
    )

    plot_rmse_rel_comparison(CELL_NAME, df_rmse_rel, models=ECM_list)

    # --- Plot parameter distributions for each (SOH, SOC) ---
    exclude_cols = {
        "trial_rank", "trial_id", "initial_guess", "estimated_params",
        "RMSE", "RMSE_rel", "RMSE_abs_relMax", "RMSE_abs_relMean", "R2_flatten", "R2_magnitude", "is_best"
    }
    for soh, n_soc in SOH_SOC_MAP.items():
        for soc in range(1, n_soc + 1):
            df_long = build_param_long_for_soh_soc(
                base_dir=BASE_DIR,
                cell=CELL_NAME,
                soh=soh,
                soc=soc,
                models=ECM_list,
                exclude_cols=exclude_cols
            )

            label = f"SOH{soh}_SOC{soc}"
            if df_long is None:
                print(f"No data collected for {label}")
                continue

            plot_param_density_grid_for_label(
                df_long=df_long,
                cell=CELL_NAME,
                label=label,
                ncols=5,        # change to 3/5 if you want
                bw_adjust=1.0   # increase for smoother, decrease for sharper
            )

