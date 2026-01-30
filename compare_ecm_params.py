import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_best_comparison(best_L, best_noL, groups):
    for group, params in groups.items():
        if not params:
            continue

        data = pd.DataFrame({
            "Parameter": params,
            "With L": [best_L[p] for p in params],
            "Without L": [best_noL[p] for p in params],
        })

        data_melt = data.melt(
            id_vars="Parameter",
            var_name="Model",
            value_name="Value"
        )

        plt.figure(figsize=(1.2 * len(params) + 2, 4))
        sns.barplot(data=data_melt, x="Parameter", y="Value", hue="Model")
        plt.title(f"Best Trial Comparison - {group} parameters")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def build_distribution_df(df, label, params):
    df_long = df[params].copy()
    df_long["Model"] = label
    return df_long.melt(
        id_vars="Model",
        var_name="Parameter",
        value_name="Value"
    )


def plot_distribution_comparison(dist_all, groups, kind="box"):
    for group, params in groups.items():
        if not params:
            continue

        subset = dist_all[dist_all["Parameter"].isin(params)]

        plt.figure(figsize=(1.2 * len(params) + 3, 5))

        if kind == "violin":
            sns.violinplot(
                data=subset,
                x="Parameter", y="Value", hue="Model",
                split=True, inner="quartile"
            )
        else:
            sns.boxplot(
                data=subset,
                x="Parameter", y="Value", hue="Model"
            )

        plt.title(f"Distribution Comparison – {group} parameters")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_rmse_distribution(df_L, df_noL):
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=pd.concat([
            df_L.assign(Model="With L")[["RMSE", "Model"]],
            df_noL.assign(Model="Without L")[["RMSE", "Model"]],
        ]),
        x="Model", y="RMSE"
    )
    plt.title("RMSE Distribution Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sns.set(style="whitegrid")

    # Load csv
    df_L = pd.read_csv("ECM_Params_Estimation/CELL042/v3CM10_RMSE_trials100/soh6/CELL042_soh6_soc1_trials100_objFunc_RMSE_v3CM10_rmOutliers2.csv")
    df_noL = pd.read_csv("ECM_Params_Estimation/CELL042/v3CM9_RMSE_trials100/soh6/CELL042_soh6_soc1_trials100_objFunc_RMSE_v3CM9_rmOutliers2.csv")

    # Shared parameters (exclude metrics & identifiers)
    exclude_cols = {
        "trial_rank", "trial_id", "initial_guess", "estimated_params",
        "RMSE", "RMSE_rel", "RMSE_abs_relMax", "RMSE_abs_relMean",
        "R2_flatten", "R2_magnitude", "is_best"
    }

    shared_params = sorted(
        set(df_L.columns)
        .intersection(df_noL.columns)
        .difference(exclude_cols)
    )

    PARAM_GROUPS = {
        "R":   [p for p in shared_params if p.startswith("R")],
        "C":   [p for p in shared_params if p.startswith("C")],
        "tau": [p for p in shared_params if p.startswith("tau")],
        "freq":[p for p in shared_params if p.startswith("freq")],
        "Aw":  ["Aw"] if "Aw" in shared_params else [],
    }

    # ---- Task 1: Best trial_rank comparison ---
    best_L = df_L.iloc[0]
    best_noL = df_noL.iloc[0]

    print(f"Best RMSE with L   : {best_L['RMSE']:.4e}")
    print(f"Best RMSE without L: {best_noL['RMSE']:.4e}")

    plot_best_comparison(best_L, best_noL, PARAM_GROUPS)

    # ---- Task 2: All trials comparison ---

    dist_L = build_distribution_df(df_L, "With L", shared_params)
    dist_noL = build_distribution_df(df_noL, "Without L", shared_params)

    dist_all = pd.concat([dist_L, dist_noL], ignore_index=True)
    plot_distribution_comparison(dist_all, PARAM_GROUPS, kind="box")

    plot_rmse_distribution(df_L, df_noL)