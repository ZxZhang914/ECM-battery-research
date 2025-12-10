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


# HelperFunc: Apply SOC-range filter logic (regular expression module)
import re 

def soc_in_range(v, soc_range):
    """
    Supports:
        - "all"
        - Single-sided:
            "Gxx", "GEQxx", "Lxx", "LEQxx"
        - Double-sided combined ranges (any order):
            "G25L40"       -> 0.25 < v < 0.40
            "GEQ20L50"     -> 0.20 <= v < 0.50
            "G10LEQ60"     -> 0.10 < v <= 0.60
            "GEQ5LEQ95"    -> 0.05 <= v <= 0.95
    """

    # Allow no filtering
    if soc_range == "all":
        return True

    # First check for combined range: e.g. G25L40, GEQ20LEQ80, ...
    m2 = re.fullmatch(
        r"(G|GEQ)(\d+(?:\.\d+)?)(L|LEQ)(\d+(?:\.\d+)?)",
        soc_range
    )
    if m2:
        lower_op, lower_val_str, upper_op, upper_val_str = m2.groups()
        lower = float(lower_val_str) / 100.0
        upper = float(upper_val_str) / 100.0

        # Lower bound check
        if lower_op == "G" and not (v > lower):
            return False
        if lower_op == "GEQ" and not (v >= lower):
            return False

        # Upper bound check
        if upper_op == "L" and not (v < upper):
            return False
        if upper_op == "LEQ" and not (v <= upper):
            return False

        return True

    # Check for single operator formats
    m1 = re.fullmatch(r"(G|GEQ|L|LEQ)(\d+(?:\.\d+)?)", soc_range)
    if m1:
        op, val_str = m1.groups()
        threshold = float(val_str) / 100.0

        if op == "G":
            return v > threshold
        if op == "GEQ":
            return v >= threshold
        if op == "L":
            return v < threshold
        if op == "LEQ":
            return v <= threshold

    # If nothing matched
    raise ValueError(
        "Invalid soc_range format. Expected 'all', Gxx, GEQxx, Lxx, LEQxx, "
        "or combined range like G25L40 or GEQ10LEQ90."
    )



def build_global_cells_df(cells, Temp_map, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats, remove_SOH=False, save_filename_prefix="fulldf_global"):
    frames = []
    for cell in cells:
        # csv_path = f"Matlab/{ECM_tag}/{cell}/{cell}_{ECM_name}_trials{num_trials}_allSOH_{stats}_{soc_range}SOC.csv"
        if remove_SOH:
            csv_path = f"ECM_Params_Estimation/{cell}/{ECM_name}_{obj_func}_trials{num_trials}/{cell}_{ECM_name}_trials{num_trials}_allSOH_remove_{stats}_{soc_range}SOC.csv"
        else:
            csv_path = f"ECM_Params_Estimation/{cell}/{ECM_name}_{obj_func}_trials{num_trials}/{cell}_{ECM_name}_trials{num_trials}_allSOH_{stats}_{soc_range}SOC.csv"

        try:
            df_i = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"[ERROR] {csv_path} not found.")
            continue

        df_i = df_i.copy()
        df_i.insert(0, "Temp", Temp_map.get(cell))
        df_i.insert(0, "CELL", cell)

        frames.append(df_i)

    df_global = pd.concat(frames, ignore_index = True)
    df_global.to_csv(f"{save_filename_prefix}_{stats}.csv")
    print(f"[SUCCESS] Combined dataframe saved as {save_filename_prefix}_{stats}.csv")

    return df_global



def build_per_cell_merged_df(cell_name, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats="all", remove_SOH=False, remove_SOHidx=[], data_source="Python"):
    celli_metadata, _ = load_cell_meta_EIS_data(cell_name)
    result_rows = []
    
    if data_source =="Matlab":
        base_folder = f"Matlab/{ECM_tag}/{cell_name}/"
    else:
        base_folder = f"ECM_Params_Estimation/{cell_name}/{ECM_name}_{obj_func}_trials{num_trials}/"
    
     # All SOH states
    for soh_i in range(celli_metadata["num_soh"]):
        if remove_SOH and ((soh_i+1) in remove_SOHidx):
            print(f"[{cell_name}] SOH {soh_i+1}, with test date {celli_metadata["soh"][soh_i]["date"]}, skipped.")
            continue
        soh_data = celli_metadata["soh"][soh_i]
        for soc_i in range(soh_data["num_soc"]):
            if data_source =="Matlab":
                if soc_i < 10:
                    soc_tag = f"0{soc_i}"
                else:
                    soc_tag = f"{soc_i}"
                file = os.path.join(
                    base_folder,
                    f"SOH{soh_i+1}/{cell_name}_SOH0{soh_i+1}_SOC{soc_tag}_{ECM_tag}.csv"
                )
            else:
                file = os.path.join(
                    base_folder,
                    f"soh{soh_i+1}/{cell_name}_soh{soh_i+1}_soc{soc_i+1}_trials{num_trials}_objFunc_{obj_func}_{ECM_name}_rmOutliers2.csv" #NOTE: Modify the path when necessary
                )
       
            df = pd.read_csv(file)
            params_names = EXPANDED_PARAMS_NAMES[ECM_name]

            # Build SOC/SOH scalars
            soh_val = soh_data["capacity"]
            soc_val = soh_data["soc"][soc_i]
            test_date = soh_data["date"]

           
            if not soc_in_range(soc_val, soc_range):
                continue

            if stats == "all":
                rows = df[params_names].copy()
                rows["date"] = test_date
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
            row_df["date"] = test_date
            row_df["SOH"] = soh_val
            row_df["SOC"] = soc_val
            result_rows.append(row_df)

    # Concatenate to build dataframe
    if result_rows:
        result_df = pd.concat(result_rows, ignore_index=True)
    else:
        params_names = EXPANDED_PARAMS_NAMES[ECM_name]
        result_df = pd.DataFrame(columns=list(params_names) + ["date", "SOH", "SOC"])

    # Save merged CSV
    if remove_SOH:
        save_path = os.path.join(
            base_folder,
            f"{cell_name}_{ECM_name}_trials{num_trials}_allSOH_remove_{stats}_{soc_range}SOC.csv",
        )
    else:
        save_path = os.path.join(
            base_folder,
            f"{cell_name}_{ECM_name}_trials{num_trials}_allSOH_{stats}_{soc_range}SOC.csv",
        )
    result_df.to_csv(save_path, index=False)
    print(f"Merged CSV files saved to {save_path}.")

    return result_df


def build_drt_merged_df(
        cells,
        Temp_map,
        soc_range="all",
        remove_SOH=False,
        remove_SOHidx=None,     # dict: cell → list of SOH indices to remove
        save_filename_prefix="fulldf_global"
    ):

    """
    Build merged DRT dataframe for all cells.

    Parameters:
        cells: list or iterable of cell names
        Temp_map: dict mapping cell → temperature
        soc_range: 'all' or one of Gxx, GEQxx, Lxx, LEQxx
        remove_SOH: bool → whether to remove certain SOH values
        remove_SOHidx: dict(cell_name → list of 1-based SOH indices to remove)
        save_filename_prefix: output CSV prefix

    Returns:
        merged_df: DataFrame containing all rows from all cells
    """

    if remove_SOH and remove_SOHidx is None:
        raise ValueError("remove_SOHidx must be provided when remove_SOH=True")

    all_data = []

    for cell_name in cells:

        drt_csv = f"../EVC_EIS_Data/CELL_DRT_Data_11-3/data_{cell_name}.csv"
        if not os.path.exists(drt_csv):
            print(f"[WARN] Missing DRT file for {cell_name}: {drt_csv}")
            continue

        # --------------------------------------------------------
        # Load and clean
        # --------------------------------------------------------
        df = pd.read_csv(drt_csv)
        df.columns = df.columns.str.strip()
        df["date"] = df["date"].astype(str).str.strip()

        # Rename columns
        df = df.rename(columns={"Charge_capacity_Ah": "SOH"})
        df = df.rename(columns={"soc": "SOC"})

        # Select needed columns
        r_cols = [c for c in df.columns if c.startswith("R")]
        freq_cols = [f"ln_1_over_freq{i}" for i in range(1, 4)]
        keep_cols = ["date", "SOC", "SOH"] + r_cols + freq_cols
        df = df[keep_cols].copy()

        # --------------------------------------------------------
        # Compute freq and tau
        # --------------------------------------------------------
        for i in range(1, 4):
            df[f"freq{i}"] = np.exp(-df[f"ln_1_over_freq{i}"])
        df.drop(columns=freq_cols, inplace=True)

        for i in range(1, 4):
            df[f"tau{i}"] = 1 / (2 * np.pi * df[f"freq{i}"])

        # --------------------------------------------------------
        # SOC filtering
        # --------------------------------------------------------
        df = df[df["SOC"].apply(lambda v: soc_in_range(v, soc_range))].reset_index(drop=True)
        
        

        # --------------------------------------------------------
        # Remove SOH (per-cell)
        # --------------------------------------------------------
        if remove_SOH:
            removed_for_this_cell = remove_SOHidx.get(cell_name, [])
            if removed_for_this_cell:
                kept_rows = []
                for i, row in df.iterrows():        # i is 0-based index
                    soh_idx = i + 1                 # convert to 1-based SOH index
                    if soh_idx in removed_for_this_cell:
                        print(f"[{cell_name}] SOH {soh_idx} removed (SOC={row['SOC']}, date={row['date']})")
                        continue
                    kept_rows.append(row)
                df = pd.DataFrame(kept_rows).reset_index(drop=True)

        # --------------------------------------------------------
        # Add cell + temp
        # --------------------------------------------------------
        df.insert(0, "Temp", Temp_map[cell_name])
        df.insert(0, "CELL", cell_name)

        # NOTE: 25 CELLS are injected with CELL042 BOL data
        # df = df[~((df["date"] == "20220916") & (df["Temp"] == 25) & (df["CELL"] != "CELL042"))]
        df = df[~((df["date"] == "20220916") & (df["CELL"] != "CELL042"))]


        all_data.append(df)

    # --------------------------------------------------------
    # Merge everything
    # --------------------------------------------------------
    merged_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    # Drop any rows containing NaN values
    merged_df = merged_df.dropna().reset_index(drop=True)

    # Save
    out_file = f"{save_filename_prefix}.csv"
    merged_df.to_csv(out_file, index=False)
    print(f"[DONE] DRT Data Processed. Saved merged merged_df: {out_file}")

    return merged_df



if __name__ == "__main__":
    
    # Configureation
    ECM_name = "v3CM9"
    ECM_tag = "ECMv9"
    obj_func = "RMSE"
    num_trials = 100
    soc_range = "G40L80" # all, "Gxx" (v > xx), "GEQxx" (v >= xx), "Lxx" (v < xx), "LEQxx" (v <= xx); where xx is an integer or float (referring to SOC percentage, e.g. soc_range = "G25")
    stats = "all"
    # ====== CONFIGURE ======
    ROOT_DIR = Path("ECM_Params_Estimation")
    #EXPECTED_COLS = ["R0","R1","R2","R3","C1","n1","C2","n2","C3","n3","Aw","SOH","SOC"]
    TEMP_MAP = {
        "CELL042": 25, "CELL050": 25, "CELL090": 25, "CELL013":25, "CELL045":25, "CELL054":25, "CELL076":25, "CELL096":25,
        "CELL009": 0,  "CELL021": 0,  "CELL077": 0,
        "CELL070": 45, "CELL101": 45, "CELL032": 45
    }
    CELLS = TEMP_MAP.keys()
    # REMOVE_DATES = { #NOTE: Remove duplicate Capacity (see summary doc table) | Remove Abnormal
    #     "CELL009": [],
    #     "CELL021": [],
    #     "CELL077": [],
    #     "CELL013": ["20240106"],
    #     "CELL042": ["20230324"],
    #     "CELL045": [],
    #     "CELL050": ["20230404"],
    #     "CELL054": ["20240108"],
    #     "CELL076": [],
    #     "CELL090": [],
    #     "CELL096": [],
    #     "CELL032": [],
    #     "CELL070": [],
    #     "CELL101": [],
    # }
    REMOVE_SOHidxS = { #NOTE: Remove duplicate Capacity [index version 1-based from high to low] (see summary doc table) | Remove Abnormal
        "CELL009": [],
        "CELL021": [],
        "CELL077": [],
        "CELL013": [2], #2
        "CELL042": [3], #3
        "CELL045": [],
        "CELL050": [2], #2
        "CELL054": [2], #2
        "CELL076": [],
        "CELL090": [1,2,3,4],
        "CELL096": [1,2,3],
        "CELL032": [],
        "CELL070": [],
        "CELL101": [],
    }
    # =======================
    DRT = False
    
    if not DRT:
        for cell in CELLS:
            build_per_cell_merged_df(cell, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats, remove_SOH=True, remove_SOHidx=REMOVE_SOHidxS[cell])
        # save_filename_prefix = f"fulldf_removeAbOod_date_{soc_range}SOC" #NOTE: Modify the path when necessary
        save_filename_prefix = f"fulldf_date_removeAbOod_{soc_range}SOC"
        build_global_cells_df(CELLS, TEMP_MAP, ECM_name, ECM_tag, obj_func, num_trials, soc_range, stats, remove_SOH=True, save_filename_prefix=save_filename_prefix)
    else:
        save_filename_prefix = f"drtdf_date_{soc_range}SOC"
        # save_filename_prefix = f"drtdf_removeAb_date_{soc_range}SOC"
        build_drt_merged_df(cells=CELLS,Temp_map=TEMP_MAP, soc_range=soc_range, remove_SOH=False, remove_SOHidx=REMOVE_SOHidxS, save_filename_prefix=save_filename_prefix)