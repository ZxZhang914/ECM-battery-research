# Sort the RC
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from Fitting_algo_v4 import *
from utils import format_EIS


# Configureation
CELL_NAME = "CELL101"
ECM_name = "v3CM9"
ECM_tag = "ECMv9"
# obj_func = "RMSE_abs"
num_trials = 50

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

    # Check ECM fitting resutls
    for soh_i in range(celli_metadata["num_soh"]):
        soh_data = celli_metadata["soh"][soh_i]
        for soc_i in range(soh_data["num_soc"]):
            print(f" ---- SOH {soh_i + 1}, SOC {soc_i + 1} ----")
            if soc_i < 10:
                soc_tag = f"0{soc_i}"
            else:
                soc_tag = f"{soc_i}"
            filepath = f"Matlab/{ECM_tag}/{CELL_NAME}/SOH{soh_i+1}/{CELL_NAME}_SOH0{soh_i+1}_SOC{soc_tag}_{ECM_tag}.csv"
            df = pd.read_csv(filepath)
            # Sort equivalent RC pairs by time constant
            for idx, row in df.iterrows():
                # Collect R, C, n, and tau values for this row
                RCn = []
                for i in range(ECM_NUM_RCS[ECM_name]):
                    R = row[f"R{i+1}"]
                    C = row[f"C{i+1}"]
                    n = row[f"n{i+1}"]
                    tau = (R * C) ** (1 / n)
                    RCn.append((tau, R, C, n))

                # Sort by tau (non-decreasing)
                RCn_sorted = sorted(RCn, key=lambda x: x[0])

                # Write sorted values back into the DataFrame
                for i, (tau, R, C, n) in enumerate(RCn_sorted, start=1):
                    df.at[idx, f"R{i}"] = R
                    df.at[idx, f"C{i}"] = C
                    df.at[idx, f"n{i}"] = n
                    df.at[idx, f"tau{i}"] = tau
            savedf_path = filepath.replace(".csv", "_sorted.csv")
            df.to_csv(savedf_path, index=False)  # Save the sorted DataFrame

if __name__ == "__main__":
    main()