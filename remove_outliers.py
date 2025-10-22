import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import seaborn as sns
import json
import glob
import random
from copy import deepcopy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, to_rgb, to_rgba

from Fitting_algo_v4 import *
from ECM_impedance_v3 import *
from utils import format_EIS, ECM_parameter_estimation

# Configureation
CELL_NAME = "CELL050"
ECM_name = "v3CM9"
ECM_tag = "ECMv9"
obj_func = "RMSE"
num_trials = 100


def main():
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

            # ---- Step 2: drop rows where ANY parameter < 1e-6
            mask_params = (df_step1[PARAMS_NAMES[ECM_name]] >= 1e-6).all(axis=1)
            df_step2 = df_step1.loc[mask_params]
            print(f"Step 2: {len(df_step2)} rows kept after filtering parameters >= 1e-6")

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

            
            # Save final output
            output_filename = filepath.replace(".csv", "_rmOutliers.csv")
            df_step5.to_csv(output_filename, index=False)
            print(f"Filtered data saved to: {output_filename}")

if __name__ == "__main__":
    main()