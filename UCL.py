# qc_ucl_hotelling.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = "ECM_Params_Estimation/CELL021/v3CM9_RMSE_trials100/soh1/"   # <--- change me: folder with one CSV per EIS
OUTPUT_DIR = "qc_plots"          # plots will be saved here
ALPHA = 0.01                     # significance level for UCL (try 0.05 if you want looser limits)
PARAM_COLS = ["R0","R1","R2","R3","C1","n1","C2","n2","C3","n3","Aw"]

# If covariance is near-singular, add a small ridge to stabilize inverse.
RIDGE_EPS = 1e-8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def hotelling_ucl(n, p, alpha=0.01):
    """
    UCL for Hotelling's T^2 when using sample covariance from n observations in p dims.
    UCL = [p*(n+1)*(n-1)/(n^2 - n*p)] * F_{alpha, p, n-p}
    """
    if n <= p:
        raise ValueError(f"Need n > p to compute Hotelling UCL (got n={n}, p={p}).")
    Fa = f.ppf(1 - alpha, dfn=p, dfd=n - p)
    return (p * (n + 1) * (n - 1) / (n**2 - n * p)) * Fa

def robust_inverse_cov(S, ridge=RIDGE_EPS):
    """
    Add a small ridge to the diagonal if needed for numerical stability.
    """
    S = np.asarray(S)
    # Ensure symmetry
    S = 0.5 * (S + S.T)
    # Add ridge if needed
    S_ridge = S + ridge * np.eye(S.shape[0])
    return np.linalg.inv(S_ridge)

def compute_hotelling_T2(X):
    """
    X: (n x p) matrix of observations (trials x parameters)
    Returns:
      T2: (n,) Hotelling T^2 values
      mu: (p,) sample mean
      S:  (p x p) sample covariance
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)
    S_inv = robust_inverse_cov(S, ridge=RIDGE_EPS)
    # Centered
    D = X - mu
    # T^2_i = d_i^T S^{-1} d_i
    T2 = np.einsum("ij,jk,ik->i", D, S_inv, D)
    return T2, mu, S

def plot_t2_chart(df, T2, UCL, eis_name, savepath):
    """
    Plot T^2 by trial with UCL line; mark OOC points.
    """
    trial_ids = df["trial_id"] if "trial_id" in df.columns else np.arange(len(T2)) + 1
    ooc = T2 > UCL

    plt.figure(figsize=(10, 5))
    plt.scatter(trial_ids, T2, marker="o", label=r"$T^2$")
    plt.axhline(UCL, linestyle="--", label=f"UCL (alpha={ALPHA})")
    # Highlight OOC
    if np.any(ooc):
        plt.scatter(np.array(trial_ids)[ooc], T2[ooc], s=80, marker="x", label="Out-of-control")
    plt.title(f"Hotelling $T^2$ Chart — {eis_name}")
    plt.xlabel("Trial")
    plt.ylabel(r"$T^2$")
    plt.ylim(-5, 65)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def plot_parameter_boxplots(df_params, eis_name, savepath, best_mask=None):
    """
    Per-parameter boxplot to see dispersion; overlay mean and (optional) best trial points.
    """
    data = [df_params[c].values for c in df_params.columns]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, showfliers=True, labels=df_params.columns, vert=True)
    # overlay parameter means
    means = [np.mean(v) for v in data]
    plt.plot(range(1, len(means) + 1), means, marker="o", linestyle="", label="Mean")
    # overlay "best" trial (if flagged)
    if best_mask is not None and best_mask.any():
        best_row = df_params.loc[best_mask].iloc[0]
        plt.plot(range(1, len(df_params.columns) + 1),
                 best_row.values, marker="x", linestyle="", label="Best trial")
    plt.title(f"Parameter Dispersion — {eis_name}")
    plt.ylabel("Parameter value")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()

def process_one_csv(path, save_plots=True, save_as_csv=True):
    fname = os.path.basename(path)
    eis_name = os.path.splitext(fname)[0]

    df = pd.read_csv(path)
    # Keep only rows with complete parameter sets
    keep = df.dropna(subset=PARAM_COLS)
    if keep.shape[0] < len(PARAM_COLS) + 1:
        print(f"[WARN] {eis_name}: not enough valid trials (n={keep.shape[0]}) for p={len(PARAM_COLS)}; skipping.")
        return

    X = keep[PARAM_COLS].to_numpy()
    n, p = X.shape

    # Compute T^2 and UCL
    try:
        T2, mu, S = compute_hotelling_T2(X)
        UCL = hotelling_ucl(n=n, p=p, alpha=ALPHA)
    except Exception as e:
        print(f"[ERROR] {eis_name}: {e}")
        return

    # Attach T^2 and OOC flags for optional downstream inspection
    keep = keep.copy()
    keep["T2"] = T2
    keep["UCL"] = UCL
    keep["is_OOC"] = keep["T2"] > keep["UCL"]

    # Save plots
    if save_plots:
        t2_path = os.path.join(OUTPUT_DIR, f"{eis_name}__T2_chart.png")
        plot_t2_chart(keep, T2=T2, UCL=UCL, eis_name=eis_name, savepath=t2_path)

        # Parameter dispersion plot
        best_mask = keep["is_best"] == True if "is_best" in keep.columns else None
        box_path = os.path.join(OUTPUT_DIR, f"{eis_name}__param_boxplots.png")
        plot_parameter_boxplots(keep[PARAM_COLS], eis_name=eis_name, savepath=box_path, best_mask=best_mask)

    # Optional: write an annotated CSV
    if save_as_csv:
        out_csv = os.path.join(OUTPUT_DIR, f"{eis_name}__qc_results.csv")
        keep.to_csv(out_csv, index=False)

    # Console summary
    num_ooc = int(keep["is_OOC"].sum())
    print(f"[OK] {eis_name}: n={n}, p={p}, UCL={UCL:.3f}, OOC trials={num_ooc}")
    
    print(keep.loc[keep["is_OOC"], "trial_id"].tolist())

def main(save_plots=True, save_as_csv=True):
    csvs = sorted(glob.glob(os.path.join(INPUT_DIR, "*_rmOutliers.csv")))
    if not csvs:
        print(f"No CSVs found in {INPUT_DIR}.")
        return
    for path in csvs:
        process_one_csv(path, save_plots=save_plots, save_as_csv=save_as_csv)

if __name__ == "__main__":
    main(save_plots=True, save_as_csv=False)
