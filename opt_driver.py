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


def main():
    parser = argparse.ArgumentParser(description="Process Args.")
    parser.add_argument("--cell_name", type=str, required=True, help="Name of the cell, e.g. CELL042")
    parser.add_argument("--ECM_name", type=str, default="v3CM9", help="ECM Name")
    parser.add_argument("--opt_method",type=str, default="LSQ", help="Optimization method, e.g. LSQ, LBFGS, Powell")
    parser.add_argument("--num_trials",type=int, default=100, help="Number of trials to run")

    args = parser.parse_args()
    CELL_NAME = args.cell_name
    ECM_name = args.ECM_name
    opt_method = args.opt_method
    num_trials = args.num_trials


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
        print(f"SOH idex {i+1}, EIS data file: {soh_data['file']}, Charge Capacity is {soh_data['capacity']}")
        # Access SOC data
        print(f"---- This SOH state has {soh_data['num_soc']} soc data: {soh_data['soc']}")


    # Build EIS dictionary
    relative_path_to_data_dir = "../EVC_EIS_Data/original_data/" # Change to match data dir
    celli_EISdata = [] # list indexed by soh


    for i in range(celli_metadata["num_soh"]):
        soh_data = celli_metadata["soh"][i]
        EIS_filename = os.path.join(relative_path_to_data_dir, soh_data["file"])
        print(f"Loading file from {EIS_filename}")
        soh_dic = format_EIS(EIS_filename)
        celli_EISdata.append(soh_dic)
    
    print(f"Optimization Algorithm {opt_method}")
    ECM_parameter_estimation(ECM_name, CELL_NAME, celli_metadata, celli_EISdata, method=opt_method, cost_func_name="RMSE", trial_number=num_trials)


if __name__ == "__main__":
    main()