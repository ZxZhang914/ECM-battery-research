#!/usr/bin/env python3
"""
Hotelling T^2 "control" style scoring:
- Use Cluster A as baseline (mean + covariance)
- Set 95% threshold (empirical percentile or parametric F-based limit)
- Score Cluster B against that threshold
- Report % of B within baseline envelope + plots

Usage:
  python t2_control.py --a clusterA.csv --b clusterB.csv
  python t2_control.py --a A.csv --b B.csv --alpha 0.05 --threshold empirical
  python t2_control.py --a A.csv --b B.csv --threshold empirical_split --split 0.7 --seed 0
  python t2_control.py --a A.csv --b B.csv --threshold parametric
  python t2_control.py --a A.csv --b B.csv --shrinkage ledoitwolf --threshold parametric
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f
from numpy.linalg import LinAlgError

ECM_COLS_DEFAULT = ["R0","R1","R2","R3","C1","C2","C3","n1","n2","n3","Aw"]


def load_csv_matrix(path: str, cols: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Available: {list(df.columns)}")

    X = df[cols].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < 3:
        raise ValueError(f"{path}: not enough valid rows after dropping NaNs.")
    return X


def mean_cov(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)
    return mu, S


def shrinkage_covariance(X: np.ndarray, method: str) -> np.ndarray:
    """
    Shrinkage covariance for stability (recommended if covariance is ill-conditioned).
    method: 'ledoitwolf' or 'oas'
    """
    method = method.lower()
    try:
        from sklearn.covariance import LedoitWolf, OAS
    except Exception as e:
        raise RuntimeError("scikit-learn required for shrinkage. Install: pip install scikit-learn") from e

    if method == "ledoitwolf":
        est = LedoitWolf().fit(X)
    elif method == "oas":
        est = OAS().fit(X)
    else:
        raise ValueError("Unknown shrinkage method. Use 'ledoitwolf' or 'oas'.")
    return est.covariance_


def mahalanobis_t2(X: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    T^2(x|A) = (x-mu)^T S^{-1} (x-mu), computed for each row x in X.
    Returns shape (n,).
    """
    try:
        Sinv = np.linalg.inv(S)
    except LinAlgError:
        raise ValueError("Covariance matrix is singular/ill-conditioned. Try --shrinkage ledoitwolf (or oas).")

    D = X - mu  # (n,p)
    # Efficient quadratic form: diag(D @ Sinv @ D^T)
    T2 = np.einsum("ij,jk,ik->i", D, Sinv, D)
    return T2


def threshold_empirical(T2_A: np.ndarray, alpha: float) -> float:
    return float(np.quantile(T2_A, 1 - alpha))


def threshold_parametric(nA: int, p: int, alpha: float) -> float:
    """
    Classic Phase II Hotelling T^2 upper control limit for a new observation
    when mu and Sigma are estimated from nA baseline samples (assumes MVN).
      h = [ p (nA+1)(nA-1) / ( nA (nA-p) ) ] * F_{p, nA-p}(1-alpha)
    Requires nA > p.
    """
    if nA <= p:
        raise ValueError(f"Parametric limit requires nA > p. Got nA={nA}, p={p}.")
    Fcrit = f.ppf(1 - alpha, p, nA - p)
    h = (p * (nA + 1) * (nA - 1) / (nA * (nA - p))) * Fcrit
    return float(h)


def plot_histograms(T2_A: np.ndarray, T2_B: np.ndarray, h: float, title: str):
    plt.figure(figsize=(10, 4))
    plt.hist(T2_A, bins=80, density=True, alpha=0.5, label="Cluster A (baseline)")
    plt.hist(T2_B, bins=80, density=True, alpha=0.5, label="Cluster B (scored vs A)")
    plt.axvline(h, linestyle="--", linewidth=2, label=f"Threshold h={h:.3g}")
    plt.xlabel("T² distance to baseline A")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV for Cluster A (baseline)")
    ap.add_argument("--b", required=True, help="CSV for Cluster B (to score)")
    ap.add_argument("--cols", nargs="*", default=ECM_COLS_DEFAULT, help="ECM columns (default: 11 ECM cols)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05 => 95th percentile)")
    ap.add_argument("--threshold", choices=["empirical", "empirical_split", "parametric"],
                    default="empirical_split",
                    help="Threshold type: empirical | empirical_split | parametric (default empirical_split)")
    ap.add_argument("--split", type=float, default=0.7,
                    help="For empirical_split: fraction of A used to fit (rest used to set percentile). Default 0.7")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (for split)")
    ap.add_argument("--shrinkage", choices=[None, "ledoitwolf", "oas"], default=None,
                    help="Optional shrinkage covariance for stability")
    ap.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = ap.parse_args()

    cols = args.cols
    alpha = args.alpha

    XA = load_csv_matrix(args.a, cols)
    XB = load_csv_matrix(args.b, cols)
    nA, p = XA.shape
    nB, p2 = XB.shape
    if p2 != p:
        raise ValueError("A and B must have the same ECM columns/dimension.")

    print(f"Loaded A: nA={nA}, B: nB={nB}, p={p}")
    print(f"Threshold mode: {args.threshold}, alpha={alpha}, shrinkage={args.shrinkage or 'none'}")

    rng = np.random.default_rng(args.seed)

    # --- Fit baseline mu,S using either all A or split portion ---
    if args.threshold == "empirical_split":
        if not (0.1 <= args.split <= 0.9):
            raise ValueError("--split should be between 0.1 and 0.9 for a meaningful split.")
        idx = rng.permutation(nA)
        n_fit = int(round(args.split * nA))
        fit_idx = idx[:n_fit]
        cal_idx = idx[n_fit:]

        A_fit = XA[fit_idx]
        A_cal = XA[cal_idx]

        muA = A_fit.mean(axis=0)
        if args.shrinkage:
            SA = shrinkage_covariance(A_fit, args.shrinkage)
        else:
            _, SA = mean_cov(A_fit)

        # T2 for calibration subset (less leakage than using same data to set percentile)
        T2_A_cal = mahalanobis_t2(A_cal, muA, SA)
        h = threshold_empirical(T2_A_cal, alpha)

        # For reporting/plots: compute T2 for all A & B using the fitted baseline
        T2_A_all = mahalanobis_t2(XA, muA, SA)
        T2_B = mahalanobis_t2(XB, muA, SA)

        title = f"T² control (baseline A fit on {n_fit}, threshold from {nA-n_fit} cal)  (alpha={alpha})"

    else:
        # Fit baseline on all A
        muA = XA.mean(axis=0)
        if args.shrinkage:
            SA = shrinkage_covariance(XA, args.shrinkage)
        else:
            _, SA = mean_cov(XA)

        T2_A_all = mahalanobis_t2(XA, muA, SA)
        T2_B = mahalanobis_t2(XB, muA, SA)

        if args.threshold == "empirical":
            h = threshold_empirical(T2_A_all, alpha)
            title = f"T² control (empirical threshold from A) (alpha={alpha})"
        elif args.threshold == "parametric":
            h = threshold_parametric(nA=nA, p=p, alpha=alpha)
            title = f"T² control (parametric F-limit) (alpha={alpha})"
        else:
            raise ValueError("Unknown threshold type.")

    # --- Report how many B are within threshold ---
    within_B = (T2_B <= h)
    frac_within = within_B.mean()
    count_within = int(within_B.sum())

    print("\n=== Baseline A (reference) ===")
    print(f"Mean vector muA (first 5): {muA[:5]}")
    print(f"Median T2(A): {np.median(T2_A_all):.6g}")
    print(f"95th pct T2(A): {np.quantile(T2_A_all, 0.95):.6g}")

    print("\n=== Threshold ===")
    print(f"h = {h:.6g}  (this corresponds to {(1-alpha)*100:.1f}th percentile envelope)")

    print("\n=== Cluster B scored vs A ===")
    print(f"Within threshold: {count_within}/{nB} = {100*frac_within:.2f}%")
    print(f"Median T2(B): {np.median(T2_B):.6g}")
    print(f"95th pct T2(B): {np.quantile(T2_B, 0.95):.6g}")
    print(f"Max T2(B): {np.max(T2_B):.6g}")

    # --- Plots ---
    if not args.no_plots:
        plot_histograms(T2_A_all, T2_B, h, title)

        # Optional: show B exceedances distribution
        plt.figure(figsize=(10, 3))
        plt.plot(np.sort(T2_B), linewidth=1.2)
        plt.axhline(h, linestyle="--", linewidth=2)
        plt.xlabel("B samples (sorted by T²)")
        plt.ylabel("T² distance to A")
        plt.title("Sorted T² for Cluster B with threshold")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
