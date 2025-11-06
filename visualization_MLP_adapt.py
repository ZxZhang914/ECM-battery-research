import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_pct_injected_effect():
    # t values (injection ratios)
    t_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3]

    # Test set results
    mape_test = [5.42, 4.46, 1.89, 2.13, 1.60, 1.77, 1.44, 1.22, 1.23, 0.82]
    r2_test = [0.3470, 0.5411, 0.9351, 0.9258, 0.9548, 0.9455, 0.9640, 0.9728, 0.9706, 0.9847]

    # Validation + Test results
    mape_valtest = [4.92, 4.11, 1.75, 2.01, 1.46, 1.63, 1.33, 1.13, 1.12, 0.75]
    r2_valtest = [0.6073, 0.7217, 0.9613, 0.9545, 0.9729, 0.9671, 0.9786, 0.9840, 0.9830, 0.9916]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(9,5))

    # --- Left Y-axis: MAPE ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Injection Ratio (t)')
    ax1.set_ylabel('MAPE (%)', color=color1)
    ax1.plot(t_values, mape_test, 'o-', color=color1, label='Test MAPE')
    ax1.plot(t_values, mape_valtest, 's--', color='dodgerblue', label='Val+Test MAPE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Right Y-axis: R² ---
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('R²', color=color2)
    ax2.plot(t_values, r2_test, 'o-', color=color2, label='Test R²')
    ax2.plot(t_values, r2_valtest, 's--', color='salmon', label='Val+Test R²')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.3, 1.0)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('SOH Prediction: MAPE and R² vs. Injection Ratio (t)')
    plt.tight_layout()
    plt.show()


def plot_by_cell():
    # Data
    # tasks = np.arange(1, 8)
    # # Test set results
    # mape_test = [4.03, 4.27, 3.97, 4.39, 3.20, 3.71, 2.86]
    # r2_test = [0.7454, 0.7557, 0.7830, 0.7430, 0.8583, 0.7026, np.nan]  # replace 0 with NaN
    # # Validation + Test results
    # mape_valtest = [3.58, 3.67, 3.33, 3.45, 2.20, 1.82, 0.63]
    # r2_valtest = [0.8448, 0.8552, 0.8718, 0.8748, 0.9384, 0.9590, 0.9903]

    # Data (old)
    # tasks = np.arange(0, 7)
    # # Test set results
    # mape_test = [24.49, 5.02, 3.91, 4.03, 3.54, 1.44, 1.50]
    # r2_test = [-10.1461, 0.5435, 0.6688, 0.5782, 0.2917, 0.9039, 0.9068]
    # # Validation + Test results
    # mape_valtest = [23.61, 4.77, 3.40, 3.05, 2.32, 0.72, 0.52]
    # r2_valtest = [-9.9405, 0.5566, 0.7371, 0.7436, 0.8019, 0.9805, 0.9888]

    # Data (NEW)
    tasks = np.arange(0, 7)
    mape_test = [13.49, 7.39, 5.55, 3.42, 2.46, 1.15, 2.31]
    r2_test   = [-2.1820, -0.5705, 0.3568, 0.6871, 0.8362, 0.9479, np.nan]

    # Validation + Test results
    mape_valtest = [13.17, 6.79, 4.75, 2.67, 1.84, 0.68, 0.58]
    r2_valtest   = [-2.1298, -0.3350, 0.5080, 0.7802, 0.8864, 0.9862, 0.9802]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(8,5))

    # Plot MAPE (left y-axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Task')
    ax1.set_ylabel('MAPE (%)', color=color1)
    ax1.plot(tasks, mape_test, 'o-', color=color1, label='Test MAPE')
    ax1.plot(tasks, mape_valtest, 's--', color='dodgerblue', label='Val+Test MAPE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(tasks)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot R² (right y-axis)
    # ax2 = ax1.twinx()
    # color2 = 'tab:red'
    # ax2.set_ylabel('R²', color=color2)
    # ax2.plot(tasks, r2_test, 'o-', color=color2, label='Test R²')
    # ax2.plot(tasks, r2_valtest, 's--', color='salmon', label='Val+Test R²')
    # ax2.tick_params(
    #     axis='y', labelcolor=color2)
    # ax2.set_ylim(0.1, 1.0)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')
    ax1.legend(lines_1 , labels_1 , loc='lower left')


    plt.title('SOH Prediction: MAPE vs Task (100% Injection Case)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_by_cell()

    # plot_pct_injected_effect()