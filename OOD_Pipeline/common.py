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


def visualize_ols_results(df, X, ols_model, plot_name_suffix=""):
    df['_pred_OLS'] = ols_model.predict(X)
    unique_cells = df['CELL'].unique()
    cmap = cm.get_cmap('tab10', len(unique_cells))
    COLOR_MAP = {cell: cmap(i) for i, cell in enumerate(unique_cells)}
    
    plt.figure(figsize=(7, 6))
    for cell in unique_cells:  
        sub = df[ df["CELL"] == cell]
        plt.scatter(
            sub["SOH"], sub["_pred_OLS"],
            alpha=0.8, s=20, label=cell, color=COLOR_MAP[cell]
        )
    # 45° reference line
    plt.plot([1.25,4.25], [1.25,4.25], "r--",  label="y=x")
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="CELLs")  # put legend outside
    plt.title(f"Predicted vs True SOH (OLS) {plot_name_suffix}")
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.axis('square')
    plt.xlim(1.25, 4.25)
    plt.ylim(1.25, 4.25)
    plt.show()


# Build Baseline LR model
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

    # Visualize results
    visualize_ols_results(df, Xz, result, plot_name_suffix=f"({model_name})")

    return {
        "name": model_name,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "standardizer": stdzr,
        "ols_result": result,
        "sigma": sigma,
        "n_train": int(len(df))
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
        return {"pct_over_3sigma": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}

    Xz = stdzr.transform(df)
    Xz = sm.add_constant(Xz)
    y = df[target_col].astype(float).to_numpy()

    y_hat = result.predict(Xz).to_numpy()
    err = y_hat - y

    return {
        "pct_over_3sigma": float(np.mean(np.abs(err) > 3.0 * sigma)),
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

