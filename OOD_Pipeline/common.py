import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Random Seed Setting
def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# Predictor Standardizer (per predictor)
@dataclass
class Standardizer:
    feature_cols: list
    mean_: pd.Series
    std_: pd.Series

    def transform(self, df):
        x = df[self.feature_cols].copy()
        std = self.std_.replace(0, 1.0) # sanitize divide by zero
        return (x - self.mean_) / std


# def visualize_ols_results(df, X, ols_model, plot_name_suffix="", save_path=None):
#     df['_pred_OLS'] = ols_model.predict(X)
#     unique_cells = df['CELL'].unique()
#     cmap = cm.get_cmap('tab10', len(unique_cells))
#     COLOR_MAP = {cell: cmap(i) for i, cell in enumerate(unique_cells)}
    
#     plt.figure(figsize=(7, 6))
#     for cell in unique_cells:  
#         sub = df[ df["CELL"] == cell]
#         plt.scatter(
#             sub["SOH"], sub["_pred_OLS"],
#             alpha=0.8, s=20, label=cell, color=COLOR_MAP[cell]
#         )
#     # 45° reference line
#     plt.plot([1.25,4.25], [1.25,4.25], "r--",  label="y=x")
#     plt.xlabel("Actual SOH")
#     plt.ylabel("Predicted SOH")
#     plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="CELLs")  # put legend outside
#     plt.title(f"Predicted vs True SOH (OLS) {plot_name_suffix}")
#     plt.grid(alpha=0.3)
#     plt.axis("equal")
#     plt.tight_layout()
#     plt.axis('square')
#     plt.xlim(1.25, 4.25)
#     plt.ylim(1.25, 4.25)
#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()



# Visualize Predicted vs True SOH using a saved payload(OLS model info)
def visualize_ols_results_payload(df, payload,
                                 plot_name_suffix="",
                                 save_path=None,
                                 axis_limits=(1.25, 4.25)):
    """
    Visualize Predicted vs True SOH using a saved payload:
      payload = {
        "target_col", "feature_cols", "standardizer", "ols_result", ...
      }

    df must contain: target column + feature columns + cell_col
    """

    # Pull schema from payload unless user overrides
    cell_col = "CELL"
    target_col = payload["target_col"]
    feature_cols = payload["feature_cols"]
    stdzr = payload["standardizer"]
    result = payload["ols_result"]

  
    need_cols = feature_cols + [target_col, cell_col]
    df_plot = df.dropna(subset=need_cols).copy()
    if df_plot.empty:
        print("No valid rows to plot after dropping NA.")
        return

    # Build standardized design matrix
    Xz = stdzr.transform(df_plot)
    Xz = sm.add_constant(Xz)
    df_plot["_pred_OLS"] = result.predict(Xz)

    # Color by cell
    unique_cells = df_plot[cell_col].astype(str).unique()
    cmap = cm.get_cmap("tab10", len(unique_cells))
    color_map = {cell: cmap(i) for i, cell in enumerate(unique_cells)}

    # Plot
    plt.figure(figsize=(7, 6))
    for cell in unique_cells:
        sub = df_plot[df_plot[cell_col].astype(str) == cell]
        plt.scatter(
            sub[target_col], sub["_pred_OLS"],
            alpha=0.8, s=20, label=cell, color=color_map[cell]
        )

    # 45-degree reference line
    lo, hi = axis_limits
    plt.plot([lo, hi], [lo, hi], "r--", label="y=x")

    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Predicted vs True {target_col} (OLS) {plot_name_suffix}")
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.axis("square")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=f"{cell_col}s")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()



# Plot Predicted vs True SOH for multiple payloads 
def visualize_multi_payload_predictions(
    df,
    payloads,
    axis_limits=(-1, 4.25),
    save_path=None,
):
    """
    Plot Predicted vs True SOH for multiple payloads.
    Each payload gets its own subplot.

    payloads: list of payload dicts
    model_names: optional list of names (same length as payloads)
    """

    if not payloads:
        raise ValueError("payloads list is empty")

    # define columns
    target_col = payloads[0]["target_col"]
    feature_cols = payloads[0]["feature_cols"]
    cell_col = "CELL"


    needed_cols = feature_cols + [target_col, cell_col]
    df_plot = df.dropna(subset=needed_cols).copy()
    if df_plot.empty:
        print("No valid rows after dropping NA.")
        return

    # Color by cell
    unique_cells = df_plot[cell_col].astype(str).unique()
    cmap = cm.get_cmap("tab10", len(unique_cells))
    color_map = {cell: cmap(i) for i, cell in enumerate(unique_cells)}

    n_models = len(payloads)
    figsize_per_plot = (6, 6)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(figsize_per_plot[0] * n_models, figsize_per_plot[1]),
        sharex=True,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    lo, hi = axis_limits

    for ax, payload, idx in zip(axes, payloads, range(n_models)):
        stdzr = payload["standardizer"]
        feature_cols = payload["feature_cols"]
        result = payload["ols_result"]
        model_name = payload["name"]

        Xz = stdzr.transform(df_plot)
        Xz = sm.add_constant(Xz, has_constant="add")
        y_hat = result.predict(Xz)

        # Scatter per cell
        for cell in unique_cells:
            sub = df_plot[df_plot[cell_col].astype(str) == cell]
            ax.scatter(
                sub[target_col],
                y_hat[sub.index],
                s=18,
                alpha=0.8,
                color=color_map[cell],
                label=cell,
            )

        # Reference line
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)

        title = f"Predicted vs True SOH (OLS) (Model: {model_name})"
        ax.set_title(title)
        ax.set_xlabel("Actual SOH")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    axes[0].set_ylabel("Predicted SOH")

    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=f"{cell_col}s",
        bbox_to_anchor=(1.02, 0.95),
        loc="upper left",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


# Build new LR model
def fit_ols_payload(df, target_col, feature_cols, model_name="model"):
    df = df.dropna(subset=feature_cols+[target_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping NA.")
    
    stdzr = Standardizer(
        feature_cols=feature_cols,
        mean_=df[feature_cols].mean(),
        std_=df[feature_cols].std(ddof=0),
    )

    Xz = stdzr.transform(df)
    Xz = sm.add_constant(Xz)
    y = df[target_col].astype(float)

    result = sm.OLS(y, Xz).fit()
    sigma = float(np.sqrt(result.mse_resid))


    return {
        "name": model_name,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "standardizer": stdzr,
        "ols_result": result,
        "sigma": sigma,
        "n_train": int(len(df)),
        "train_mape": float(np.mean(np.abs((result.predict(Xz) - y) / y))) * 100.0,
        "train_mae": float(np.mean(np.abs(result.predict(Xz) - y))),
        "train_rmse": float(np.sqrt(np.mean((result.predict(Xz) - y) ** 2)))
    }

# evaluate df OLS fitting using saved payload data
def eval_payload_on_df(payload, df):
    target_col = payload["target_col"]
    feature_cols = payload["feature_cols"]
    stdzr = payload["standardizer"]
    result = payload["ols_result"]
    sigma = float(payload["sigma"])

    df = df.dropna(subset=feature_cols+[target_col]).reset_index(drop=True)
    if df.empty:
        return {"pct_over_3sigma": np.nan, "mape": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}

    Xz = stdzr.transform(df)
    Xz = sm.add_constant(Xz)
    y = df[target_col].astype(float).to_numpy()

    y_hat = result.predict(Xz).to_numpy()
    err = y_hat - y

    return {
        "pct_over_3sigma": float(np.mean(np.abs(err) > 3.0 * sigma)),
        "mape": float(np.mean(np.abs(err / y))) * 100.0,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": int(len(y)),
    }


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

