import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [

# ======================================================
# Test 1
# ======================================================
# LR
{"TestID": 1, "Model": "LR",  "Cell": "CELL013", "MAPE_mean": 3.53, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL042", "MAPE_mean": 3.32, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL045", "MAPE_mean": 1.67, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL050", "MAPE_mean": 1.08, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL054", "MAPE_mean": 1.48, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL076", "MAPE_mean": 0.95, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL090", "MAPE_mean": 4.45, "MAPE_std": np.nan},
{"TestID": 1, "Model": "LR",  "Cell": "CELL096", "MAPE_mean": 2.19, "MAPE_std": np.nan},

# DNN
{"TestID": 1, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 4.00, "MAPE_std": 1.62},
{"TestID": 1, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 3.87, "MAPE_std": 0.65},
{"TestID": 1, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 3.67, "MAPE_std": 0.56},
{"TestID": 1, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 2.64, "MAPE_std": 0.16},
{"TestID": 1, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 0.87, "MAPE_std": 0.39},
{"TestID": 1, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 0.88, "MAPE_std": 0.26},
{"TestID": 1, "Model": "DNN", "Cell": "CELL090", "MAPE_mean": 2.79, "MAPE_std": 1.20},
{"TestID": 1, "Model": "DNN", "Cell": "CELL096", "MAPE_mean": 1.95, "MAPE_std": 0.20},


# ======================================================
# Test 2 — 8 cells LOOCV with partial SOC
# ======================================================
# LR
{"TestID": 2, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 2.93, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.06, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.44, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 1.96, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 1.96, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.48, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL090", "MAPE_mean": 4.12, "MAPE_std": np.nan},
{"TestID": 2, "Model": "LR", "Cell": "CELL096", "MAPE_mean": 1.97, "MAPE_std": np.nan},

# DNN
{"TestID": 2, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 4.17, "MAPE_std": 2.91},
{"TestID": 2, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 3.21, "MAPE_std": 0.36},
{"TestID": 2, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 3.58, "MAPE_std": 0.24},
{"TestID": 2, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 3.98, "MAPE_std": 0.60},
{"TestID": 2, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 1.13, "MAPE_std": 0.28},
{"TestID": 2, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 1.51, "MAPE_std": 0.65},
{"TestID": 2, "Model": "DNN", "Cell": "CELL090", "MAPE_mean": 3.35, "MAPE_std": 0.99},
{"TestID": 2, "Model": "DNN", "Cell": "CELL096", "MAPE_mean": 2.44, "MAPE_std": 1.16},


# ======================================================
# Test 3 — 6 cells, remove CELL090 & CELL096
# ======================================================
# LR
{"TestID": 3, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 2.77, "MAPE_std": np.nan},
{"TestID": 3, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.19, "MAPE_std": np.nan},
{"TestID": 3, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.47, "MAPE_std": np.nan},
{"TestID": 3, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 3.22, "MAPE_std": np.nan},
{"TestID": 3, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 2.94, "MAPE_std": np.nan},
{"TestID": 3, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.90, "MAPE_std": np.nan},

# DNN
{"TestID": 3, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 2.57, "MAPE_std": 1.44},
{"TestID": 3, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 3.60, "MAPE_std": 0.59},
{"TestID": 3, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 2.54, "MAPE_std": 0.68},
{"TestID": 3, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 3.93, "MAPE_std": 0.66},
{"TestID": 3, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 2.03, "MAPE_std": 0.81},
{"TestID": 3, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 0.76, "MAPE_std": 0.37},


# ======================================================
# Test 4 — 6 cells + partial SOC
# ======================================================
# LR
{"TestID": 4, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 3.02, "MAPE_std": np.nan},
{"TestID": 4, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.73, "MAPE_std": np.nan},
{"TestID": 4, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 2.22, "MAPE_std": np.nan},
{"TestID": 4, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 3.37, "MAPE_std": np.nan},
{"TestID": 4, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 2.59, "MAPE_std": np.nan},
{"TestID": 4, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.34, "MAPE_std": np.nan},

# DNN
{"TestID": 4, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 2.56, "MAPE_std": 1.33},
{"TestID": 4, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 4.30, "MAPE_std": 1.18},
{"TestID": 4, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 2.80, "MAPE_std": 0.71},
{"TestID": 4, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 2.47, "MAPE_std": 1.05},
{"TestID": 4, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 5.95, "MAPE_std": 2.32},
{"TestID": 4, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 1.56, "MAPE_std": 0.50},


# ======================================================
# Test 5 — [SOC>40%] 6 cells, remove 090/096 + inconsistent + OOD
# ======================================================
# LR
{"TestID": 5, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 1.98, "MAPE_std": np.nan},
{"TestID": 5, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.61, "MAPE_std": np.nan},
{"TestID": 5, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.35, "MAPE_std": np.nan},
{"TestID": 5, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 2.94, "MAPE_std": np.nan},
{"TestID": 5, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 0.88, "MAPE_std": np.nan},
{"TestID": 5, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.81, "MAPE_std": np.nan},

# DNN
{"TestID": 5, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 2.09, "MAPE_std": 0.90},
{"TestID": 5, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 12.02, "MAPE_std": 5.45},
{"TestID": 5, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 3.81, "MAPE_std": 0.84},
{"TestID": 5, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 3.86, "MAPE_std": 0.49},
{"TestID": 5, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 1.03, "MAPE_std": 0.22},
{"TestID": 5, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 1.13, "MAPE_std": 0.59},


# ======================================================
# Test 6 — [SOC>40%] 6 cells + partial SOC (same filtering as 5)
# ======================================================
# LR
{"TestID": 6, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 2.13, "MAPE_std": np.nan},
{"TestID": 6, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.53, "MAPE_std": np.nan},
{"TestID": 6, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.58, "MAPE_std": np.nan},
{"TestID": 6, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 3.68, "MAPE_std": np.nan},
{"TestID": 6, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 1.14, "MAPE_std": np.nan},
{"TestID": 6, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 1.44, "MAPE_std": np.nan},

# DNN
{"TestID": 6, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 2.48, "MAPE_std": 0.09},
{"TestID": 6, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 26.08, "MAPE_std": 5.19},
{"TestID": 6, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 3.95, "MAPE_std": 1.79},
{"TestID": 6, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 14.26, "MAPE_std": 3.33},
{"TestID": 6, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 1.41, "MAPE_std": 0.53},
{"TestID": 6, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 3.79, "MAPE_std": 2.26},


# ======================================================
# Test 7 — [SOC>40%, no SOC as predictor] 6 cells
# ======================================================
# LR
{"TestID": 7, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 2.25, "MAPE_std": np.nan},
{"TestID": 7, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.98, "MAPE_std": np.nan},
{"TestID": 7, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.44, "MAPE_std": np.nan},
{"TestID": 7, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 3.03, "MAPE_std": np.nan},
{"TestID": 7, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 0.51, "MAPE_std": np.nan},
{"TestID": 7, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.67, "MAPE_std": np.nan},

# DNN
{"TestID": 7, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 1.60, "MAPE_std": 0.45},
{"TestID": 7, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 11.30, "MAPE_std": 3.46},
{"TestID": 7, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 3.69, "MAPE_std": 0.88},
{"TestID": 7, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 3.85, "MAPE_std": 1.35},
{"TestID": 7, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 0.77, "MAPE_std": 0.21},
{"TestID": 7, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 1.35, "MAPE_std": 1.14},

# ======================================================
# Test 8 — [SOC>40%, no SOC as predictor] 6 cells + partial SOC
# ======================================================
# LR
{"TestID": 8, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 1.93, "MAPE_std": np.nan},
{"TestID": 8, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 3.80, "MAPE_std": np.nan},
{"TestID": 8, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.56, "MAPE_std": np.nan},
{"TestID": 8, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 3.65, "MAPE_std": np.nan},
{"TestID": 8, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 0.42, "MAPE_std": np.nan},
{"TestID": 8, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.92, "MAPE_std": np.nan},

# DNN
{"TestID": 8, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 1.17, "MAPE_std": 0.30},
{"TestID": 8, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 53.42, "MAPE_std": 6.78},
{"TestID": 8, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 3.77, "MAPE_std": 1.02},
{"TestID": 8, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 18.30, "MAPE_std": 0.83},
{"TestID": 8, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 1.92, "MAPE_std": 1.41},
{"TestID": 8, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 2.12, "MAPE_std": 1.13},

# ======================================================
# Test 9 — [DRT SOC>25%] 8 cells LOOCV (no SOC as predictor)
# ======================================================

# LR
{"TestID": 9, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 3.18, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 2.99, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 0.90, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 2.30, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 0.62, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.19, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL090", "MAPE_mean": 1.48, "MAPE_std": np.nan},
{"TestID": 9, "Model": "LR", "Cell": "CELL096", "MAPE_mean": 2.08, "MAPE_std": np.nan},

# DNN (mean; std)
{"TestID": 9, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 4.93, "MAPE_std": 1.30},
{"TestID": 9, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 3.79, "MAPE_std": 2.02},
{"TestID": 9, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 1.21, "MAPE_std": 0.82},
{"TestID": 9, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 5.68, "MAPE_std": 0.87},
{"TestID": 9, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 7.15, "MAPE_std": 2.98},
{"TestID": 9, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 1.33, "MAPE_std": 0.41},
{"TestID": 9, "Model": "DNN", "Cell": "CELL090", "MAPE_mean": 4.81, "MAPE_std": 2.08},
{"TestID": 9, "Model": "DNN", "Cell": "CELL096", "MAPE_mean": 2.35, "MAPE_std": 0.56},


# ======================================================
# Test 10 — [DRT SOC>25%] 8 cells LOOCV with partial SOC (no SOC as predictor)
# ======================================================

# LR
{"TestID": 10, "Model": "LR", "Cell": "CELL013", "MAPE_mean": 3.62, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL042", "MAPE_mean": 2.93, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL045", "MAPE_mean": 1.19, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL050", "MAPE_mean": 2.47, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL054", "MAPE_mean": 1.03, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL076", "MAPE_mean": 0.20, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL090", "MAPE_mean": 1.77, "MAPE_std": np.nan},
{"TestID": 10, "Model": "LR", "Cell": "CELL096", "MAPE_mean": 1.91, "MAPE_std": np.nan},

# DNN
{"TestID": 10, "Model": "DNN", "Cell": "CELL013", "MAPE_mean": 6.45, "MAPE_std": 1.66},
{"TestID": 10, "Model": "DNN", "Cell": "CELL042", "MAPE_mean": 4.50, "MAPE_std": 1.47},
{"TestID": 10, "Model": "DNN", "Cell": "CELL045", "MAPE_mean": 1.48, "MAPE_std": 0.38},
{"TestID": 10, "Model": "DNN", "Cell": "CELL050", "MAPE_mean": 6.75, "MAPE_std": 0.77},
{"TestID": 10, "Model": "DNN", "Cell": "CELL054", "MAPE_mean": 7.08, "MAPE_std": 2.77},
{"TestID": 10, "Model": "DNN", "Cell": "CELL076", "MAPE_mean": 1.62, "MAPE_std": 0.89},
{"TestID": 10, "Model": "DNN", "Cell": "CELL090", "MAPE_mean": 6.76, "MAPE_std": 4.13},
{"TestID": 10, "Model": "DNN", "Cell": "CELL096", "MAPE_mean": 2.45, "MAPE_std": 0.79},

]

df = pd.DataFrame(data)
print(df.head())


# 1) consistent order for cells & TestIDs
cells = sorted(df["Cell"].unique())
test_ids = sorted(df["TestID"].unique())

x = np.arange(len(cells))          # base x-locations for cell groups
n_tests = len(test_ids)
offset_width = 0.6 / n_tests       # how much to shift each test's dots

# color per TestID
color_map = {tid: f"C{i}" for i, tid in enumerate(test_ids)}

plt.figure(figsize=(14, 6))

test_ids = [9,10]

for i, tid in enumerate(test_ids):
    df_t = df[df["TestID"] == tid]

    # compute horizontal offset for this test
    offset = (i - (n_tests - 1) / 2) * offset_width

    # ----- LR (just dots) -----
    df_lr = df_t[df_t["Model"] == "LR"].set_index("Cell").reindex(cells)
    lr_mean = df_lr["MAPE_mean"].to_numpy(dtype=float)
    mask_lr = ~np.isnan(lr_mean)

    plt.scatter(
        x[mask_lr] + offset,
        lr_mean[mask_lr],
        color=color_map[tid],
        marker="D",
        s=40,
        edgecolors="black",
        linewidths=0.6,
        # label=f"Test {tid} - LR"
    )

    # ----- DNN (dot + error bar) -----
    df_dnn = df_t[df_t["Model"] == "DNN"].set_index("Cell").reindex(cells)
    dnn_mean = df_dnn["MAPE_mean"].to_numpy(dtype=float)
    dnn_std  = df_dnn["MAPE_std"].to_numpy(dtype=float)
    mask_dnn = ~np.isnan(dnn_mean)

    plt.errorbar(
        x[mask_dnn] + offset,
        dnn_mean[mask_dnn],
        yerr=dnn_std[mask_dnn],
        fmt="o",
        color=color_map[tid],
        ecolor=color_map[tid],
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        label=f"Test {tid} - DNN"
    )

# 2) Formatting
plt.xticks(x, cells)
plt.ylabel("MAPE (%)")
plt.xlabel("Test Cell")
plt.title("MAPE per Cell: LR (dimond) and DNN (dot ± std) across Test IDs")
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Legend: only show LR and DNN once
handles, labels = plt.gca().get_legend_handles_labels()
seen = {}
ordered_handles = []
ordered_labels = []

for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = True
        ordered_handles.append(h)
        ordered_labels.append(l)

plt.legend(ordered_handles, ordered_labels, loc="upper right")

plt.tight_layout()

# dynamic y-limit
max_val = (df["MAPE_mean"] + df["MAPE_std"].fillna(0)).max()
plt.ylim(0, max_val * 1.2)
plt.ylim(0, 14)
plt.show()



# ensure cell order is consistent on x-axis
cells = sorted(df["Cell"].unique())

# color per TestID
test_ids = sorted(df["TestID"].unique())
color_map = {
    tid: f"C{i}" for i, tid in enumerate(test_ids)
}  # uses matplotlib default colors C0,C1,...

plt.figure(figsize=(10, 6))

for tid in test_ids:
    df_t = df[df["TestID"] == tid]

    # ----- LR (solid line) -----
    df_lr = df_t[df_t["Model"] == "LR"].set_index("Cell").reindex(cells)
    if not df_lr.empty:
        plt.plot(
            cells,
            df_lr["MAPE_mean"],
            marker="o",
            linestyle="-",
            color=color_map[tid],
            label=f"Test {tid} - LR",
        )

    # ----- DNN (dashed line + std bar) -----
    df_dnn = df_t[df_t["Model"] == "DNN"].set_index("Cell").reindex(cells)
    if not df_dnn.empty:
        y = df_dnn["MAPE_mean"].values
        yerr = df_dnn["MAPE_std"].values

        # use errorbar for std; dashed line
        plt.errorbar(
            cells,
            y,
            yerr=yerr,
            fmt="--o",             # dashed line with circle markers
            color=color_map[tid],  # same color as LR for this test
            ecolor=color_map[tid],
            elinewidth=1.2,
            capsize=3,
            label=f"Test {tid} - DNN",
        )

plt.ylabel("MAPE (%)")
plt.xlabel("Test Cell")
plt.title("MAPE comparison across tests (LR vs DNN)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
# plt.ylim(0, 5)
plt.show()


########## -----------------------Boxplot Visualization----------------------- ##########
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df must contain: TestID, Model, Cell, MAPE_mean, MAPE_std
# (from previous code you built)
# -------------------------------------------------------------

test_ids = sorted(df["TestID"].unique())
test_ids = [9, 10]

plt.figure(figsize=(14, 6))

# We create one subplot per test
n_tests = len(test_ids)
fig, axes = plt.subplots(1, n_tests, figsize=(4*n_tests, 6), sharey=True)

if n_tests == 1:  # If only one test, axes is not an array
    axes = [axes]

for ax, tid in zip(axes, test_ids):
    
    df_t = df[df["TestID"] == tid]

    # ---------- DNN data ----------
    dnn = df_t[df_t["Model"] == "DNN"]
    dnn_values = dnn["MAPE_mean"].values


    # Boxplot for DNN
    bp = ax.boxplot(
        dnn_values,
        positions=[1],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue'),
        medianprops=dict(color='black'),
    )

    # ---------- LR data ----------
    lr = df_t[df_t["Model"] == "LR"]
    lr_values = lr["MAPE_mean"].values


    # Plot LR mean as a single dot
    # ax.scatter(
    #     [1.6],
    #     [lr_mean],
    #     color="red",
    #     s=80,
    #     label="LR mean"
    # )
    bp = ax.boxplot(
        lr_values,
        positions=[1.6],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor='lightpink', color='pink'),
        medianprops=dict(color='black'),
    )


    # ---------- Labels ----------
    ax.set_title(f"Test {tid}", fontsize=14)
    ax.set_xticks([1, 1.6])
    ax.set_xticklabels(["DNN\n", "LR\n"])
    ax.set_ylim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.5)

# Only one legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=12)

fig.suptitle("MAPE Comparison per Test: DNN Boxplot vs LR Mean Dot", fontsize=16)
plt.tight_layout(rect=[0.03,0.03,0.93,0.95])
plt.show()
