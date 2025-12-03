#!/bin/bash

# ----- User configuration -----
PYTHON_SCRIPT_MLP="MLPRegressor_loocv.py"
PYTHON_SCRIPT_LR="LR_loocv.py"

# ----- Command arguments -----
FEATURES=("R0" "R1" "R2" "R3" "SOC")
INPUT_CSV="fulldf_date_G40SOC_all.csv"

# ----- Output directories -----
OUTPUT_DIR_MLP="MLP_plots/LOOCV_TEST_MLP"
OUTPUT_DIR_LR="MLP_plots/LOOCV_TEST_LR"

# Create directories
mkdir -p "$OUTPUT_DIR_MLP"
mkdir -p "$OUTPUT_DIR_LR"

LOG_MLP="${OUTPUT_DIR_MLP}/log.txt"
LOG_LR="${OUTPUT_DIR_LR}/log.txt"

# Convert array to CLI-friendly string
FEATURES_ARG=""
for f in "${FEATURES[@]}"; do
    FEATURES_ARG="$FEATURES_ARG $f"
done


# ============================================================
# 1. Run MLP LOOCV
# ============================================================

echo "Running MLP LOOCV: $PYTHON_SCRIPT_MLP"
echo "Logging to $LOG_MLP"

python "$PYTHON_SCRIPT_MLP" \
    --features $FEATURES_ARG \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_DIR_MLP" \
    > "$LOG_MLP" 2>&1

echo "MLP run complete."


# ============================================================
# 2. Run LR LOOCV
# ============================================================

echo "Running LR LOOCV: $PYTHON_SCRIPT_LR"
echo "Logging to $LOG_LR"

python "$PYTHON_SCRIPT_LR" \
    --features $FEATURES_ARG \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_DIR_LR" \
    > "$LOG_LR" 2>&1

echo "LR run complete."
echo "Done."
