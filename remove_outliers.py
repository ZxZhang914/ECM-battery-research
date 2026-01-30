import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import seaborn as sns
import json
import glob
import random
import argparse
from copy import deepcopy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, to_rgb, to_rgba

from Fitting_algo_v4 import *
from ECM_impedance_v3 import *
from utils import format_EIS, ECM_parameter_estimation

# This script removes outliers from ECM parameter estimation results. To run this script, make sure you have ECM parameter estimation CSV files available.
# based on multiple criteria including frequency limits, parameter bounds, rmse thresholds, r2 thresholds, and optional percentile-based filtering.

# Example usage:
# python remove_outliers.py --cell-name CELL042 --ecm-name v3CM10 --obj-func RMSE --num-trials 100 --remove-based-on-parameters-percentile --pct-central 0.95 --eps-params 1e-6 --eps-L 1e-9


def main():
    parser = argparse.ArgumentParser(description="Process Args.")
    parser.add_argument("--cell-name", type=str, default="CELL090",help="Cell name identifier")
    parser.add_argument("--ecm-name", type=str, default="v3CM9",help="ECM model name")
    parser.add_argument("--obj-func", type=str, default="RMSE",choices=["RMSE", "MAE", "MSE"],help="Objective function")
    parser.add_argument("--num-trials", type=int, default=100,help="Number of optimization trials")
    parser.add_argument("--remove-based-on-parameters-percentile",action="store_true",help="Apply parameter percentile filtering (criterion 6)")
    parser.add_argument("--pct-central", type=float, default=0.95, help="Central percentile for parameter filtering")
    parser.add_argument("--eps-params", type=float, default=1e-6,help="Minimum value for parameters (except L)")
    parser.add_argument("--eps-L", type=float, default=1e-9,help="Minimum value for parameter L")
    args = parser.parse_args()

    CELL_NAME = args.cell_name
    ECM_name = args.ecm_name
    obj_func = args.obj_func
    num_trials = args.num_trials
    remove_based_on_parameters_percentile = args.remove_based_on_parameters_percentile
    pct_central = args.pct_central
    eps_params = args.eps_params
    eps_L = args.eps_L

    # ==== Load battery metadata from JSON file ====
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json" # Check the path

    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)   # <--- this is now a dict

    # ==== Load metadata ====
    celli_metadata = battery_metadata[CELL_NAME]
    print(celli_metadata["temperature"])
    print(celli_metadata["num_soh"])

    # Build EIS dictionary
    relative_path_to_data_dir = "../EVC_EIS_Data/original_data/" # Change to match data dir
    celli_EISdata = [] # list indexed by soh


    for i in range(celli_metadata["num_soh"]):
        soh_data = celli_metadata["soh"][i]
        EIS_filename = os.path.join(relative_path_to_data_dir, soh_data["file"])
        print(f"Loading file from {EIS_filename}")
        soh_dic = format_EIS(EIS_filename)
        celli_EISdata.append(soh_dic)
    
    for soh_i in range(celli_metadata["num_soh"]):
        soh_data = celli_metadata["soh"][soh_i]
        for soc_i in range(soh_data["num_soc"]):
            filepath = f"ECM_Params_Estimation/{CELL_NAME}/{ECM_name}_{obj_func}_trials{num_trials}/soh{soh_i+1}/{CELL_NAME}_soh{soh_i+1}_soc{soc_i+1}_trials{num_trials}_objFunc_{obj_func}_{ECM_name}.csv"
            df = pd.read_csv(filepath)
            df["RMSE_abs_relMax"] = df["RMSE"] / (celli_EISdata[soh_i][soc_i]["Z_mag"].max())
            df["RMSE_abs_relMean"] = df["RMSE"] / (celli_EISdata[soh_i][soc_i]["Z_mag"].mean())

            # ---- Step 1: drop rows where freq1 > 10K
            mask_freq1 = df["freq1"] <= 10000
            df_step1 = df.loc[mask_freq1]
            print(f"Step 1: {len(df_step1)} rows kept after freq1 <= 10kHz")

            # ---- Step 2: drop rows where ANY parameter < eps_params (or eps_L for L)
            param_names = PARAMS_NAMES[ECM_name]
            mask_params = df_step1[param_names].ge(eps_params)
            mask_params["L"] = df_step1["L"].ge(eps_L)
            df_step2 = df_step1.loc[mask_params.all(axis=1)]
            print(f"Step 2: {len(df_step2)} rows kept after filtering parameters")

            # ---- Step 3: drop rows where RMSE_relMean> 0.015
            if "RMSE_abs_relMean" in df_step2.columns:
                mask_rmse = df_step2["RMSE_abs_relMean"] <= 0.015
                df_step3 = df_step2.loc[mask_rmse]
                if len(df_step3) < 10:
                    df_step3 = df_step2
                    print(f"Step 3: less than 10 rows kept after RMSE_abs <= 0.015, skip this cirterion")
                print(f"Step 3: {len(df_step3)} rows kept after RMSE_abs <= 0.015")
            else:
                print("Warning: 'RMSE_abs' column not found. Skipping Step 3.")
                df_step3 = df_step2

            # ---- Step 4: drop rows where R2_magnitude < 0.998
            if "R2_magnitude" in df_step3.columns:
                mask_r2 = df_step3["R2_magnitude"] >= 0.998
                df_step4 = df_step3.loc[mask_r2]
                if len(df_step4) < 10:
                    df_step4 = df_step3
                    print(f"Step 4: less than 10 rows kept after R2_magnitude >= 0.998, skip this cirterion")
                print(f"Step 4: {len(df_step4)} rows kept after R2_magnitude >= 0.998")
            else:
                print("Warning: 'R2_magnitude' column not found. Skipping Step 4.")
                df_step4 = df_step3
            
            # ---- Step 5: keep rows within the 95th percentile of error
            # Prefer RMSE_abs_relMean; fallback to RMSE if unavailable
            if "RMSE_abs_relMean" in df_step4.columns:
                q95 = df_step4["RMSE_abs_relMean"].quantile(0.95)
                mask_p95 = df_step4["RMSE_abs_relMean"] <= q95
                df_step5 = df_step4.loc[mask_p95]
                metric_used = "RMSE_abs_relMean"
            elif "RMSE" in df_step4.columns:
                q95 = df_step4["RMSE"].quantile(0.95)
                mask_p95 = df_step4["RMSE"] <= q95
                df_step5 = df_step4.loc[mask_p95]
                metric_used = "RMSE"
            else:
                print("Warning: neither 'RMSE_abs_relMean' nor 'RMSE' found. Skipping Step 5.")
                df_step5 = df_step4
                metric_used = None

            if metric_used is not None:
                if len(df_step5) < 10:
                    df_step5 = df_step4
                    print(f"Step 5: less than 10 rows kept after 95th percentile on {metric_used} (q95={q95:.6g}), skip this criterion")
                else:
                    print(f"Step 5: {len(df_step5)} rows kept after 95th percentile on {metric_used} (q95={q95:.6g})")
            else:
                print("Step 5: skipped (no error metric)")

            final_df = df_step5
            
            if remove_based_on_parameters_percentile:
                # NOTE: This step is optional 
                # ---- Step 6: keep rows that all parameters within central 95th percentile
                lw_pct = (1 - pct_central) / 2
                up_pct = 1 - lw_pct
                df_step6 = df_step5.copy()
                for param in PARAMS_NAMES[ECM_name]:
                    lower_bound = df_step5[param].quantile(lw_pct)
                    upper_bound = df_step5[param].quantile(up_pct)
                    mask_param = (df_step5[param] >= lower_bound) & (df_step5[param] <= upper_bound)
                    df_step6 = df_step6.loc[mask_param]
                    if len(df_step6) < 10:
                        df_step6 = df_step5
                        print(f"Step 6: less than 10 rows kept after filtering {param} within central {pct_central}, skip this criterion")
                    else:
                        print(f"Step 6: {len(df_step6)} rows kept after filtering {param} within central {pct_central} percentile")
                final_df = df_step6
            
            # Save final output #NOTE: change filename here if needed
            output_filename = filepath.replace(".csv", "_rmOutliers2.csv")
            final_df.to_csv(output_filename, index=False)
            print(f"Filtered data saved to: {output_filename}")

if __name__ == "__main__":
    main()