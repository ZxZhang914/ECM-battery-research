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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.backends.backend_pdf import PdfPages

from Fitting_algo_v4 import *
from ECM_impedance_v3 import *

# ==== Globals ====
ECM_candidates_map = dict(zip(ECM_NAMES, ECM_IMPEDANCE_FUNCS))

# ==== Functions
def format_EIS(input_file):
    """
    Load EIS input csv and return a list of dictionary indexed by soc_index.

    Parameters
    ----------
    input_file : str
        csv file path of original EIS data, required to have columns ['time/s', 'I/mA', 'Ewe/V', 'freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg', 'z cycle'].
    
    Returns
    ----------
    grouped_list : [{soc_dict}], a list of SOC dictionaries. soc_dict has attributes: freq, angular_freq, Z_mag, phase_deg, Z_real, Z_imag.
    """
    df = pd.read_csv(input_file)

    # Ensure relevant columns exist
    required = ['time/s', 'I/mA', 'Ewe/V', 'freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg', 'z cycle']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric
    num_cols = ['time/s', 'I/mA', 'Ewe/V', 'freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg', 'z cycle']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep relevant cols and drop invalid rows
    df = (
        df[['freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg', 'z cycle']]
        .dropna(subset=['freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg', 'z cycle'])
        .query("`freq/Hz` != 0")
        .copy()
    )

    # Compute Z components
    phase_rad = np.deg2rad(df['Phase(Z)/deg'].to_numpy())
    Z_mag = df['|Z|/Ohm'].to_numpy()
    df['Z_mag'] = Z_mag
    df['Z_real'] = Z_mag * np.cos(phase_rad)
    df['Z_imag'] = Z_mag * np.sin(phase_rad)
    df['angular_freq'] = 2 * np.pi * df['freq/Hz'].to_numpy()
    df['soc_index'] = df['z cycle'].astype(int)

    # Group by z cycle and build list (index = soc_index)
    grouped_list = []
    for _, g in df.groupby('z cycle'):
        g_sorted = g.sort_values('freq/Hz', ascending=False)
        grouped_list.append({
            "freq": g_sorted['freq/Hz'].to_numpy(dtype=float),
            "angular_freq": g_sorted['angular_freq'].to_numpy(dtype=float),
            "Z_mag": g_sorted['Z_mag'].to_numpy(dtype=float),
            "phase_deg": g_sorted['Phase(Z)/deg'].to_numpy(dtype=float),
            "Z_real": g_sorted['Z_real'].to_numpy(dtype=float),
            "Z_imag": g_sorted['Z_imag'].to_numpy(dtype=float),
        })

    return grouped_list



def EIS_Nyquist_meas_vs_fit_save(
    battery_name, battery_metadata, battery_EISdata,
    soh_label, soc_label, ECM_name, est_result_file,
    select_est_ranks=[1], subtitle="", show_legend=True
):
    soh_soc_eis = battery_EISdata[soh_label-1][soc_label-1]
    Z_meas_real = soh_soc_eis["Z_real"]
    Z_meas_negimag = -soh_soc_eis["Z_imag"]
    angular_freq = soh_soc_eis["angular_freq"]

    if os.path.exists(est_result_file):
        df = pd.read_csv(est_result_file)
        # selected = df[df["RunNo"].isin(select_est_ranks)].copy()
        selected = df[df["trial_id"].isin(select_est_ranks)].copy()

    else:
        print(f"[WARN] est_result_file not found: {est_result_file}")
        return None

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(Z_meas_real, Z_meas_negimag, marker="o", label="Lab Measurement")

    for _, row in selected.iterrows():
        # rank = int(row["RunNo"])
        rank = int(row["trial_id"])
        est_params = row[PARAMS_NAMES[ECM_name]].tolist()
        Z_fit = ECM_candidates_map[ECM_name](est_params, angular_freq)
        ax.plot(Z_fit.real, -Z_fit.imag, linestyle="--", marker="*", label=f"ECM fit (rank={rank})")

    ax.set_xlabel("Re(Z) [Ω]")
    ax.set_ylabel("-Im(Z) [Ω]")
    if show_legend:
        ax.legend()

    cap = battery_metadata["soh"][soh_label - 1]["capacity"]
    soc_val = battery_metadata["soh"][soh_label - 1]["soc"][soc_label - 1] * 100.0
    ax.set_title(
        f"{battery_name} Lab EIS vs ECM Reconstruction\n"
        f"SOH_index = {soh_label}; Capacity = {cap:.2f}\n"
        f"SOC_index = {soc_label}; SOC = {soc_val:.2f}%\n"
        f"{subtitle} \n"
        f"n={len(selected)}"
    )

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def tau_vec(R, Q, n):
    """
    Compute time constant (tau) for R-CPE component; For vector input.

    Paramters
    ----------
    R : list of float
        Resistance.
    
    Q : list of float
        Capacitance
    
    n : list of float
        Ideal factor
    
    Returns
    ----------
    Time constant value as vector
    """
    R = R.astype(float).to_numpy()
    Q = Q.astype(float).to_numpy()
    n = n.astype(float).to_numpy()
    out = np.full(R.shape, np.nan, dtype=float)  # default to NaN
    valid = (R > 0) & (Q > 0) & (n > 0) & np.isfinite(R) & np.isfinite(Q) & np.isfinite(n)
    out[valid] = np.power(R[valid] * Q[valid], 1.0 / n[valid])
    return out


def ECM_parameter_estimation(ECM_model_name, target_battery_name, target_battery_metadata, target_battery_EISdata, method="LSQ", cost_func_name="RMSE", trial_number=50):
    print(f"ECM Parameter Estimation for {target_battery_name}")
    
    # All SOH states
    for soh_i in range(target_battery_metadata["num_soh"]):
        # if soh_i !=0: continue #
        soh_data = target_battery_metadata["soh"][soh_i]
        # print(f"Hardcode for CELL 50: i={soh_i}, capacity={soh_data["capacity"]}, num_soc={soh_data["num_soc"]}") #
        
        # All SOC states
        for soc_i in range(soh_data["num_soc"]):
            print(f"--- SOH = ({soh_i + 1}), Capacity = {soh_data["capacity"]}; SOC = ({soc_i+1}), {soh_data["soc"][soc_i]*100:.2f}%")
            soc_EISdata = target_battery_EISdata[soh_i][soc_i]
            Z_meas = soc_EISdata["Z_real"] + soc_EISdata["Z_imag"]*1j
            angular_freq = soc_EISdata["angular_freq"]

            # find ECM candidates
            if ECM_model_name not in ECM_candidates_map:
                print(f"[Warning] ECM_model_name '{ECM_model_name}' not found. Use 'v3CM7' as default."
                    f"Available options: {list(ECM_candidates_map.keys())}")
                ECM_model_name = "v3CM7"
            ECM_candidate_impedance_func = ECM_candidates_map[ECM_model_name]
            
            # call fitting function
            # LSQ no need cost_func_name and optimizer_option
            best_params, best_err, err_str, trial_results = ECM_result_wrapper_v4(Z_meas, angular_freq,
                ECM_model_name, ECM_candidate_impedance_func, 
                trial_num=trial_number, ECM_initial_guess=None, ECM_bounds=None, 
                cost_func_name=cost_func_name, verbose=False, method=method, optimizer_option=None
            )

            df = pd.DataFrame(trial_results)
            # Optional: Format Original Dataframe

            param_col_names = PARAMS_NAMES[ECM_model_name]
            print(param_col_names)
            params_df = pd.DataFrame(df["estimated_params"].tolist(), columns=param_col_names, index=df.index)
            
            freq_df = pd.DataFrame()
            for i in range(ECM_NUM_RCS[ECM_model_name]):
                
                tau_i = tau_vec(params_df[f"R{i+1}"], params_df[f"C{i+1}"], params_df[f"n{i+1}"])
                params_df[f"tau{i+1}"] = tau_i
                freq_df[f"freq{i+1}"] = 1.0 / (2.0 * np.pi * tau_i)
            
            df_out = pd.concat([df, params_df, freq_df], axis=1)
            
            # Sort rows by the cost function column
            df_out = df_out.sort_values(by=[f"{cost_func_name}"], ascending=True).reset_index(drop=True)
            df_out.insert(0, "trial_rank", range(1, len(df_out) + 1))

            save_dir = f"ECM_Params_Estimation/{target_battery_name}/{ECM_model_name}_{cost_func_name}_trials{trial_number}/soh{soh_i+1}/"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{target_battery_name}_soh{soh_i+1}_soc{soc_i+1}_trials{trial_number}_objFunc_{cost_func_name}_{ECM_model_name}.csv")
            df_out.to_csv(save_path, index=False)
            print("--- CSV file saved.")



def load_cell_meta_EIS_data(cell_name):
    # ==== Load battery metadata from JSON file ====
    battery_json_file = "../EVC_EIS_Data/original_data/Battery_Info_DRT.json" # Check the path

    with open(battery_json_file, "r") as f:
        battery_metadata = json.load(f)   # <--- this is now a dict
    
    # ==== Load metadata ====
    celli_metadata = battery_metadata[cell_name]
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
    
    return celli_metadata, celli_EISdata


def build_cell_colormap(metadata, shade_min=0.30, shade_max=0.90, base_maps=None):
    """
    Create a dict: {cell_name: '#RRGGBB'}.
    
    Parameters
    ----------
    metadata : dict
        loaded from json file
    shade_min, shade_max : float in [0,1]
        Range within the colormap to sample shades (avoid extremes that are too light/dark).
    base_maps : dict or None
        Optional override of base colormaps per temperature, e.g. {0:"Blues", 25:"Greens", 45:"Reds"}.
    """
    # Default temperature → base colormap
    default_maps = {0: "Blues", 25: "Greens", 45: "Reds"}
    if base_maps:
        default_maps.update(base_maps)
    
    # Group cells by temperature
    groups = {}
    for cell, info in metadata.items():
        t = int(info["temperature"])
        groups.setdefault(t, []).append(cell)
    
    # Build color map per group, assigning distinct shades
    cell_to_color = {}
    for t, cells in groups.items():
        cells_sorted = sorted(cells)  # deterministic assignment
        n = len(cells_sorted)
        if n == 1:
            positions = [0.6]
        else:
            positions = np.linspace(shade_min, shade_max, n)
        
        cmap_name = default_maps.get(t, None)
        if cmap_name is None or cmap_name not in mpl.colormaps:
            # Fallback if an unexpected temperature shows up
            cmap = mpl.colormaps["gray"]
        else:
            cmap = mpl.colormaps[cmap_name]
        
        for cell, pos in zip(cells_sorted, positions):
            rgba = cmap(pos)
            cell_to_color[cell] = mpl.colors.to_hex(rgba)
    
    return cell_to_color

