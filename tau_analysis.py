import os
import glob
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "ECM_Params_Estimation/CELL090/v3CM8_RMSE_trials50"

JSON_FILE = "/home/warrenzzx/projects/EVC_EIS_Data/original_data/Battery_Info_DRT.json"

CELL_NAME = "CELL090"

# ------------------------------------------------
# 读取 JSON metadata
# ------------------------------------------------

with open(JSON_FILE, "r") as f:
    metadata = json.load(f)

cell_info = metadata[CELL_NAME]

# build lookup
soh_capacity = {}
soc_lookup = {}

for i, soh in enumerate(cell_info["soh"], start=1):

    capacity = soh["capacity"]
    soh_capacity[i] = capacity

    for j, soc in enumerate(soh["soc"], start=1):
        soc_lookup[(i, j)] = soc

print("Loaded metadata")

# ------------------------------------------------
# 读取 ECM fitting CSV
# ------------------------------------------------

rows = []

for file in glob.glob(BASE_DIR + "/**/*.csv", recursive=True):

    filename = os.path.basename(file)

    soh_group = int(filename.split("_soh")[1].split("_")[0])
    soc_group = int(filename.split("_soc")[1].split("_")[0])

    df = pd.read_csv(file)

    # 选 best trial
    if "is_best" in df.columns:
        df = df[df["is_best"] == True]
    else:
        df = df.sort_values("RMSE").iloc[[0]]

    capacity = soh_capacity[soh_group]
    soc = soc_lookup[(soh_group, soc_group)]

    df["capacity"] = capacity
    df["soc"] = soc

    rows.append(df)

df_all = pd.concat(rows, ignore_index=True)

print("Loaded EIS points:", len(df_all))

# ------------------------------------------------
# log tau
# ------------------------------------------------

for i in [1,2,3]:

    if f"tau{i}" in df_all.columns:

        df_all[f"log_tau{i}"] = np.log10(df_all[f"tau{i}"])
        df_all[f"log_freq{i}"] = np.log10(df_all[f"freq{i}"])

df_all["soc_percent"] = df_all["soc"] * 100

# ------------------------------------------------
# tau heatmap
# ------------------------------------------------

pivot = df_all.pivot_table(
    index="capacity",
    columns="soc_percent",
    values="log_tau1"
)

pivot = pivot.sort_index().sort_index(axis=1)

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis")
plt.title("log(tau1)")
plt.xlabel("SOC (%)")
plt.ylabel("Capacity (Ah)")
plt.tight_layout()
plt.savefig("tau1_heatmap.png", dpi=300)
plt.show()

# ------------------------------------------------
# tau boxplot
# ------------------------------------------------

tau_cols = [c for c in df_all.columns if "log_tau" in c]

df_tau = df_all.melt(
    id_vars=["capacity"],
    value_vars=tau_cols,
    var_name="Tau_Type",
    value_name="LogTau"
)

plt.figure(figsize=(8,6))
sns.boxplot(data=df_tau, x="capacity", y="LogTau", hue="Tau_Type")
plt.title("Tau vs Capacity")
plt.xlabel("Capacity (Ah)")
plt.ylabel("log10(tau)")
plt.tight_layout()
plt.savefig("tau_boxplot.png", dpi=300)
plt.show()

# ------------------------------------------------
# freq heatmap
# ------------------------------------------------

pivot = df_all.pivot_table(
    index="capacity",
    columns="soc_percent",
    values="log_freq1"
)

pivot = pivot.sort_index().sort_index(axis=1)

plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="viridis")
plt.title("log(freq1)")
plt.xlabel("SOC (%)")
plt.ylabel("Capacity (Ah)")
plt.tight_layout()
plt.savefig("freq_heatmap.png", dpi=300)
plt.show()