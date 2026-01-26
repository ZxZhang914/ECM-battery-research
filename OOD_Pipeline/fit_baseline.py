import os
import json
import argparse
import pandas as pd

from common import fit_ols_payload, save_pickle, set_global_seed

#### Example Usage ####
'''
python fit_baseline.py --csv ../fulldf_date_removeAbOod_G40L80SOC_all.csv --target SOH --outdir artifacts

'''

def main():
    set_global_seed(42)

    parser = argparse.ArgumentParser(description="Fit baseline OLS model.")
    parser.add_argument("--csv", required=True, help="Baseline CSV file") # Currently pass in all cell data and later filter 25degree cells
    parser.add_argument("--target", required=True, help="Target column (e.g. SOH)")
    parser.add_argument("--features", default="R0, R1, R2, R3", help="Comma-separated feature list")
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--soc-min", type=float, default=0.4,
                    help="Minimum SOC (inclusive), float number")
    parser.add_argument("--soc-max", type=float, default=0.8,
                    help="Maximum SOC (inclusive), float number")
    args = parser.parse_args()


    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # filter SOC range
    print(f"Filtering SOC between {args.soc_min} and {args.soc_max}")
    df = df[(df["SOC"] >= args.soc_min) & (df["SOC"] <= args.soc_max)]
    # Note [temp]: filter 25oc normal cells
    # selected_cells = ["CELL042", "CELL050", "CELL013", "CELL045", "CELL054", "CELL076"]
    df = df[df["Temp"] == 25]


    if args.features.strip():
        features_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        print("Invalid features, use R0, R1, R2, R3 instead.")
        features_cols = ["R0", "R1", "R2", "R3"]
    
    payload = fit_ols_payload(df, args.target, features_cols, model_name = args.name)

    pkl_path = os.path.join(args.outdir, "baseline_model.pkl")
    meta_path = os.path.join(args.outdir, "baseline_model_meta.json")

    # Save pickle
    save_pickle(payload, pkl_path)
    meta = {
        "baseline_csv": args.csv,
        "name": payload["name"],
        "target_col": payload["target_col"],
        "feature_cols": payload["feature_cols"],
        "soc_min": args.soc_min,
        "soc_max": args.soc_max,
        "sigma": payload["sigma"],
        "n_train": payload["n_train"],
        "r2": payload["ols_result"].rsquared,
        "mae": float(payload["ols_result"].resid.abs().mean()),
        "rmse": float((payload["ols_result"].resid ** 2).mean() ** 0.5),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved baseline model to {pkl_path} and metadata to {meta_path}")
    print(payload["ols_result"].summary())


if __name__ == "__main__":
    main()