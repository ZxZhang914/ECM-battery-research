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
from matplotlib.backends.backend_pdf import PdfPages

from Fitting_algo_v4 import *
from ECM_impedance_v3 import *
from utils import format_EIS, EIS_Nyquist_meas_vs_fit_save

# TJhis script generates a PDF file containing Nyquist plots comparing measured EIS data with ECM fitting results across multiple SOH and SOC states for a specified battery cell and ECM model.
# To run this script, ensure you have the necessary ECM fitting result CSV files available.

# Example usage:
# python reconstruction_visualization.py --cell-name CELL042 --ecm-name v3CM10  --obj-func RMSE --num-trials 100

def main():

    parser = argparse.ArgumentParser(description="ECM result configuration")
    parser.add_argument("--cell-name", type=str, default="CELL013", help="Cell name identifier")
    parser.add_argument("--ecm-name", type=str, default="v3CM9", help="ECM model name")
    parser.add_argument("--ecm-tag", type=str, default="ECMv9", help="ECM tag") # this is for matlab result
    parser.add_argument("--obj-func", type=str, default="RMSE", choices=["RMSE", "MAE", "MSE"], help="Objective function")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--pdf-path", type=str, default=None, help="Output PDF path (auto-generated if not provided)")
    args = parser.parse_args()

    CELL_NAME = args.cell_name
    ECM_name = args.ecm_name
    ECM_tag = args.ecm_tag
    obj_func = args.obj_func
    num_trials = args.num_trials
    pdf_path = args.pdf_path
    if pdf_path is None:
        pdf_path = f"PythonResult_{CELL_NAME}_{ECM_name}_trials{num_trials}.pdf"

    # ==== Load battery metadata from JSON file ====
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json" # Check the path

    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)   # <--- this is now a dict

    # ==== Load metadata ====
    celli_metadata = battery_metadata[CELL_NAME]
    print(celli_metadata["temperature"])
    print(celli_metadata["num_soh"])   

    # Battery data retrieval example
    for i in range(celli_metadata["num_soh"]):
        # Access SOH data
        soh_data = celli_metadata["soh"][i]
        print(f"SOH idex {i+1}, EIS data file: {soh_data["file"]}, Charge Capacity is {soh_data["capacity"]}")
        # Access SOC data
        print(f"---- This SOH state has {soh_data["num_soc"]} soc data: {soh_data["soc"]}")


    # Build EIS dictionary
    relative_path_to_data_dir = "../EVC_EIS_Data/original_data/" # Change to match data dir
    celli_EISdata = [] # list indexed by soh


    for i in range(celli_metadata["num_soh"]):
        soh_data = celli_metadata["soh"][i]
        EIS_filename = os.path.join(relative_path_to_data_dir, soh_data["file"])
        print(f"Loading file from {EIS_filename}")
        soh_dic = format_EIS(EIS_filename)
        celli_EISdata.append(soh_dic)


    # ECM Fitting Results
    with PdfPages(pdf_path) as pdf:
        for soh_i in range(celli_metadata["num_soh"]):
            soh_data = celli_metadata["soh"][soh_i]
            for soc_i in range(soh_data["num_soc"]):
                print(f" ---- SOH {soh_i + 1}, SOC {soc_i + 1} ----")
                if soc_i < 10:
                    soc_tag = f"0{soc_i}"
                else:
                    soc_tag = f"{soc_i}"
                # NOTE: Hardcoded filepath pattern, modify if necessary
                # filepath = f"Matlab/{ECM_tag}/{CELL_NAME}/SOH{soh_i+1}/{CELL_NAME}_SOH0{soh_i+1}_SOC{soc_tag}_{ECM_tag}_sorted.csv"
                filepath = f"ECM_Params_Estimation/{CELL_NAME}/{ECM_name}_{obj_func}_trials{num_trials}/soh{soh_i+1}/{CELL_NAME}_soh{soh_i+1}_soc{soc_i+1}_trials{num_trials}_objFunc_{obj_func}_{ECM_name}_rmOutliers2.csv"
                subtitle = f"trials{num_trials}_{ECM_name}"

                df = pd.read_csv(filepath)
                # select_ranks = df["RunNo"].values
                select_ranks = df["trial_id"].values
                # select_ranks = [1,22,33,]
                df["RMSE_abs_relMax"] = df["RMSE"] / (celli_EISdata[soh_i][soc_i]["Z_mag"].max())
                df["RMSE_abs_relMean"] = df["RMSE"] / (celli_EISdata[soh_i][soc_i]["Z_mag"].mean())
                
                stats1 = (
                    "RMSE (relative to max Z) stats -> "
                    + f"mean={df['RMSE_abs_relMax'].mean():.4f}, "
                    + f"median={df['RMSE_abs_relMax'].median():.4f}, "
                    + f"max={df['RMSE_abs_relMax'].max():.4f}, "
                    + f"min={df['RMSE_abs_relMax'].min():.4f}"
                )

                stats2 = (
                    "RMSE (relative to mean Z) stats -> "
                    + f"mean={df['RMSE_abs_relMean'].mean():.4f}, "
                    + f"median={df['RMSE_abs_relMean'].median():.4f}, "
                    + f"max={df['RMSE_abs_relMean'].max():.4f}, "
                    + f"min={df['RMSE_abs_relMean'].min():.4f}"
                )

                subtitle = subtitle + "\n" + stats1 + "\n" + stats2

                fig = EIS_Nyquist_meas_vs_fit_save(CELL_NAME, celli_metadata, celli_EISdata, soh_i+1, soc_i+1, ECM_name, filepath, select_est_ranks=select_ranks, subtitle=subtitle, show_legend=False)
                if fig:
                    pdf.savefig(fig, bbox_inches="tight")
                    print("saving")
                    plt.close(fig)   # close to save memory
                

    print(f"All figures saved to: {pdf_path}")

if __name__ == "__main__":
    main()