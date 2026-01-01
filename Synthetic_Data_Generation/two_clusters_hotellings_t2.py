#!/usr/bin/env python3
"""
Two-sample Hotelling's T^2 test for comparing mean ECM vectors between Cluster A and B.

Usage examples:
  python two_clusters_hotellings_t2.py --a clusterA.csv --b clusterB.csv
  python two_clusters_hotellings_t2.py --a clusterA.csv --b clusterB.csv --shrinkage ledoitwolf
  python two_clusters_hotellings_t2.py --a clusterA.csv --b clusterB.csv --perm 5000 --seed 0
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f
from numpy.linalg import LinAlgError

ECM_COLS_DEFAULT = ["R0","R1","R2","R3","C1","C2","C3","n1","n2","n3","Aw"]


def load_cluster(csv_path: str, cols: list[str]) -> np.ndarray:
    df = pd.read_csv(csv_path)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}\n"
                         f"Available columns: {list(df.columns)}")

    X = df[cols].to_numpy(dtype=float)

    # Drop rows with any NaNs
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]

    if X.shape[0] < 2:
        raise ValueError(f"{csv_path}: not enough valid rows after dropping NaNs.")

    return X


def sample_mean_cov(X: np.ndarray):
    """
    Returns:
      mean: (p,)
      cov:  (p,p) unbiased sample covariance (ddof=1)
    """
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    return mean, cov


def pooled_cov(SA: np.ndarray, SB: np.ndarray, nA: int, nB: int) -> np.ndarray:
    return ((nA - 1) * SA + (nB - 1) * SB) / (nA + nB - 2)


def hotellings_t2_two_sample(XA: np.ndarray, XB: np.ndarray, use_shrinkage: str | None = None):
    """
    Two-sample Hotelling's T^2 using pooled covariance.

    If use_shrinkage is provided, we estimate pooled covariance using shrinkage on the
    combined centered data for numerical stability.

    Returns dict with T2, F, df1, df2, pvalue, and diagnostics.
    """
    nA, p = XA.shape
    nB, p2 = XB.shape
    if p2 != p:
        raise ValueError("XA and XB must have the same number of columns.")

    xbarA, SA = sample_mean_cov(XA)
    xbarB, SB = sample_mean_cov(XB)
    delta = (xbarA - xbarB).reshape(-1, 1)  # (p,1)

    N = nA + nB
    if N - 2 <= p:
        raise ValueError(
            f"Need N-2 > p to use standard pooled-covariance Hotelling's T^2.\n"
            f"Got N={N}, p={p}. Consider shrinkage or permutation."
        )

    if use_shrinkage is None:
        Sp = pooled_cov(SA, SB, nA, nB)
    else:
        # Shrinkage covariance on combined centered data (practical, stable)
        Z = np.vstack([XA - xbarA, XB - xbarB])  # centered within group
        Sp = shrinkage_cov(Z, method=use_shrinkage)

    # T^2 = (nA*nB/(nA+nB)) * delta^T Sp^{-1} delta
    try:
        Sp_inv = np.linalg.inv(Sp)
    except LinAlgError:
        raise ValueError(
            "Pooled covariance is singular/ill-conditioned. "
            "Try --shrinkage ledoitwolf or --shrinkage oas, or use --perm."
        )

    scale = (nA * nB) / (nA + nB)
    T2 = float(scale * (delta.T @ Sp_inv @ delta))

    # Convert to F
    df1 = p
    df2 = N - p - 1
    Fstat = (df2 / (df1 * (N - 2))) * T2  # = (N-p-1)/(p(N-2)) * T2
    pval = 1.0 - f.cdf(Fstat, df1, df2)

    return {
        "nA": nA, "nB": nB, "p": p,
        "meanA": xbarA, "meanB": xbarB,
        "delta": (xbarA - xbarB),
        "T2": T2,
        "F": float(Fstat),
        "df1": int(df1),
        "df2": int(df2),
        "pvalue": float(pval),
        "cov_used": use_shrinkage or "pooled (unshrunk)"
    }


def shrinkage_cov(Z: np.ndarray, method: str) -> np.ndarray:
    """
    Shrinkage covariance estimator.
    method: 'ledoitwolf' or 'oas'
    """
    method = method.lower()
    try:
        from sklearn.covariance import LedoitWolf, OAS
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for shrinkage covariance. "
            "Install with: pip install scikit-learn"
        ) from e

    if method == "ledoitwolf":
        est = LedoitWolf().fit(Z)
    elif method == "oas":
        est = OAS().fit(Z)
    else:
        raise ValueError("Unknown shrinkage method. Use 'ledoitwolf' or 'oas'.")

    return est.covariance_


def permutation_pvalue(XA: np.ndarray, XB: np.ndarray, B: int = 5000, seed: int = 0,
                       use_shrinkage: str | None = None):
    """
    Permutation p-value for T^2.
    Recomputes T^2 under random label shuffles.

    Returns (p_perm, T2_obs).
    """
    rng = np.random.default_rng(seed)

    # Observed
    obs = hotellings_t2_two_sample(XA, XB, use_shrinkage=use_shrinkage)
    T2_obs = obs["T2"]

    X = np.vstack([XA, XB])
    nA = XA.shape[0]
    nB = XB.shape[0]
    N = nA + nB

    count = 0
    for _ in range(B):
        perm = rng.permutation(N)
        idxA = perm[:nA]
        idxB = perm[nA:]
        XA_p = X[idxA, :]
        XB_p = X[idxB, :]
        try:
            T2_p = hotellings_t2_two_sample(XA_p, XB_p, use_shrinkage=use_shrinkage)["T2"]
        except ValueError:
            # If singular in some permuted split (rare), skip that draw
            continue
        if T2_p >= T2_obs:
            count += 1

    # +1 smoothing for exactness
    p_perm = (count + 1) / (B + 1)
    return float(p_perm), float(T2_obs)


def format_vec(cols, v):
    return "\n".join([f"  {c:>3s}: {v[i]: .6g}" for i, c in enumerate(cols)])


def plot_t2_projection(XA, XB, Sp):
    """
    Visualize Hotelling T^2 via Fisher/LDA projection.
    XA, XB: (nA,p), (nB,p)
    Sp: pooled covariance (p,p)
    """
    muA = XA.mean(axis=0)
    muB = XB.mean(axis=0)

    # Fisher / T^2 direction
    w = np.linalg.solve(Sp, muA - muB)
    w = w / np.linalg.norm(w)

    sA = XA @ w
    sB = XB @ w

    plt.figure(figsize=(8,4))
    plt.hist(sA, bins=60, density=True, alpha=0.6, label="Cluster A")
    plt.hist(sB, bins=60, density=True, alpha=0.6, label="Cluster B")
    plt.axvline(sA.mean(), linestyle="--")
    plt.axvline(sB.mean(), linestyle="--")
    plt.xlabel("T² / LDA projection score")
    plt.ylabel("Density")
    plt.title("Hotelling T² (Fisher) Projection")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV for cluster A")
    ap.add_argument("--b", required=True, help="CSV for cluster B")
    ap.add_argument("--cols", nargs="*", default=ECM_COLS_DEFAULT,
                    help="ECM columns to use (default: 11 ECM columns)")
    ap.add_argument("--shrinkage", default=None, choices=[None, "ledoitwolf", "oas"],
                    help="Use shrinkage covariance for stability")
    ap.add_argument("--perm", type=int, default=0,
                    help="If >0, compute permutation p-value with this many permutations")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for permutation test")
    args = ap.parse_args()

    cols = args.cols
    XA = load_cluster(args.a, cols)
    XB = load_cluster(args.b, cols)

    # Param checks / quick summary
    print(f"Loaded A: {XA.shape[0]} rows, B: {XB.shape[0]} rows, p={XA.shape[1]}")
    print(f"Columns: {cols}")

    res = hotellings_t2_two_sample(XA, XB, use_shrinkage=args.shrinkage)

    print("\n=== Two-sample Hotelling's T^2 ===")
    print(f"Covariance used: {res['cov_used']}")
    print(f"T^2  = {res['T2']:.6g}")
    print(f"F    = {res['F']:.6g}  with df=({res['df1']},{res['df2']})")
    print(f"pval = {res['pvalue']:.6g}")

    print("\n--- Means ---")
    print("Mean(A):\n" + format_vec(cols, res["meanA"]))
    print("Mean(B):\n" + format_vec(cols, res["meanB"]))
    print("Delta (A-B):\n" + format_vec(cols, res["delta"]))

    if args.perm and args.perm > 0:
        p_perm, T2_obs = permutation_pvalue(
            XA, XB, B=args.perm, seed=args.seed, use_shrinkage=args.shrinkage
        )
        print("\n=== Permutation test (label shuffle) ===")
        print(f"Permutations: {args.perm}, seed={args.seed}")
        print(f"T^2 (obs) = {T2_obs:.6g}")
        print(f"p_perm    = {p_perm:.6g}")

    # Optional: interpret quickly
    alpha = 0.05
    decision = "REJECT H0 (means differ)" if res["pvalue"] < alpha else "FAIL TO REJECT H0"
    print(f"\nDecision at alpha={alpha}: {decision}")

    # Visualize T^2 projection
    plot_t2_projection(XA, XB, pooled_cov(
        *sample_mean_cov(XA)[1:],
        *sample_mean_cov(XB)[1:],
        XA.shape[0], XB.shape[0]
    ))



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
