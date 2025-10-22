import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import seaborn as sns
import json
import glob
import random
from pathlib import Path

from Fitting_algo_v4 import *
from utils import load_cell_meta_EIS_data

# For matlab version

def build_global_cells_df(cells, Temp_map, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats):
    frames = []
    for cell in cells:
        # csv_path = f"Matlab/{ECM_tag}/{cell}/{cell}_{ECM_name}_trials{num_trials}_allSOH_{stats}_{soc_range}SOC.csv"
        csv_path = f"ECM_Params_Estimation/{cell}/{ECM_name}_{obj_func}_trials{num_trials}/{cell}_{ECM_name}_trials{num_trials}_allSOH_{stats}_{soc_range}SOC.csv"

        if csv_path is None:
            print(f"[ERROR] {csv_path} not found.")
        df_i = pd.read_csv(csv_path)

        df_i = df_i.copy()
        df_i.insert(0, "Temp", Temp_map.get(cell))
        df_i.insert(0, "CELL", cell)

        frames.append(df_i)

    df_global = pd.concat(frames, ignore_index = True)
    df_global.to_csv(f"fulldf_global_{stats}.csv")



def build_per_cell_merged_df_matlab(cell_name, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats="all"):
    celli_metadata, _ = load_cell_meta_EIS_data(cell_name)
    result_rows = []
    base_folder = f"ECM_Params_Estimation/{cell_name}/{ECM_name}_{obj_func}_trials{num_trials}/"
    # base_folder = f"Matlab/{ECM_tag}/{cell_name}/"
    # if suffix:
    #     base_folder = f"ECM_Params_Estimation/{battery_name}/{ECM_model_name}_{obj_func}_trials{num_trials}_{suffix}/"
    
     # All SOH states
    for soh_i in range(celli_metadata["num_soh"]):
        soh_data = celli_metadata["soh"][soh_i]
        for soc_i in range(soh_data["num_soc"]):
            if soc_i < 10:
                soc_tag = f"0{soc_i}"
            else:
                soc_tag = f"{soc_i}"
            # file = os.path.join(
            #     base_folder,
            #     f"SOH{soh_i+1}/{cell_name}_SOH0{soh_i+1}_SOC{soc_tag}_{ECM_tag}.csv"
            # )
            file = os.path.join(
                base_folder,
                f"soh{soh_i+1}/{cell_name}_soh{soh_i+1}_soc{soc_i+1}_trials{num_trials}_objFunc_{obj_func}_{ECM_name}_rmOutliers.csv"
            )
       
            df = pd.read_csv(file)
            params_names = PARAMS_NAMES[ECM_name]

            # Build SOC/SOH scalars
            soh_val = soh_data["capacity"]
            soc_val = soh_data["soc"][soc_i]

            # Apply SOC-range filter logic
            def soc_in_range(v):
                if soc_range == "all":
                    return True
                if soc_range == "G25":     # > 0.25
                    return v > 0.25
                if soc_range == "LEQ25":   # <= 0.25
                    return v <= 0.25
                raise ValueError("soc_range must be one of ['all', 'G25', 'LEQ25']")

            if not soc_in_range(soc_val):
                continue

            if stats == "all":
                rows = df[params_names].copy()
                rows["SOH"] = soh_val
                rows["SOC"] = soc_val
                result_rows.append(rows)
                continue

            # Single-row stats
            if stats == "median":
                row = df[params_names].median()
            elif stats == "mean":
                row = df[params_names].mean()
            elif stats == "best":
                # df is already ordered so that the first row is "best"
                row = df[params_names].iloc[0]
            else:
                raise ValueError("stats must be one of ['median', 'mean', 'best', 'all']")

            row_df = row.to_frame().T  # normalize to a 1-row DataFrame
            row_df["SOH"] = soh_val
            row_df["SOC"] = soc_val
            result_rows.append(row_df)

    # Concatenate to build dataframe
    if result_rows:
        result_df = pd.concat(result_rows, ignore_index=True)
    else:
        params_names = PARAMS_NAMES[ECM_name]
        result_df = pd.DataFrame(columns=list(params_names) + ["SOH", "SOC"])

    # Save merged CSV
    save_path = os.path.join(
        base_folder,
        f"{cell_name}_{ECM_name}_trials{num_trials}_allSOH_{stats}_{soc_range}SOC.csv",
    )
    result_df.to_csv(save_path, index=False)
    print(f"Merged CSV files saved to {save_path}.")

    return result_df




if __name__ == "__main__":
    
    # Configureation
    ECM_name = "v3CM9"
    ECM_tag = "ECMv9"
    obj_func = "RMSE"
    num_trials = 100
    soc_range = "G25"
    stats = "median"
    # ====== CONFIGURE ======
    ROOT_DIR = Path("ECM_Params_Estimation")
    #EXPECTED_COLS = ["R0","R1","R2","R3","C1","n1","C2","n2","C3","n3","Aw","SOH","SOC"]
    # TEMP_MAP = {
    #     "CELL042": 25, "CELL050": 25, "CELL090": 25,
    #     "CELL009": 0,  "CELL021": 0,  "CELL077": 0,
    #     "CELL070": 45, "CELL101": 45, "CELL032": 45
    # }
    TEMP_MAP = {
        "CELL042": 25, "CELL050": 25, "CELL090": 25, "CELL013":25, "CELL045":25, "CELL054":25, "CELL076":25, "CELL096":25,
        "CELL009": 0,  "CELL021": 0,  "CELL077": 0,
        "CELL070": 45, "CELL101": 45, "CELL032": 45
    }
    CELLS = TEMP_MAP.keys()
    # =======================
    for cell in CELLS:
        build_per_cell_merged_df_matlab(cell, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats)
    build_global_cells_df(CELLS, TEMP_MAP, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats)

    