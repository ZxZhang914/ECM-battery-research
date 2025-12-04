#!/bin/bash

# ----- User configuration -----
PYTHON_SCRIPT_MLP="MLPRegressor_loocv.py"
PYTHON_SCRIPT_LR="LR_loocv.py"

# Toggle: true => add --partial_SOC and add suffix
USE_PARTIAL_SOC=false


# ----- Command arguments -----
FEATURES=("R0" "R1" "R2" "R3") # Note: Change as needed
INPUT_CSV="fulldf_date_removeAbOod_G40SOC_all.csv"


# ----- Build suffix based on toggle -----
SUFFIX=""
PARTIAL_SOC_FLAG=""

if [ "$USE_PARTIAL_SOC" = true ]; then
    SUFFIX="_partialSOC"
    PARTIAL_SOC_FLAG="--partial_SOC"
fi


# ----- Output directories with suffix -----
OUTPUT_DIR_MLP="MLP_plots/LOOCV_MLP${SUFFIX}"
OUTPUT_DIR_LR="MLP_plots/LOOCV_LR${SUFFIX}"

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
echo "Saving results to: $OUTPUT_DIR_MLP"
echo "Logging to: $LOG_MLP"

python "$PYTHON_SCRIPT_MLP" \
    --features $FEATURES_ARG \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_DIR_MLP" \
    $PARTIAL_SOC_FLAG \
    > "$LOG_MLP" 2>&1

echo "MLP run complete."


# ============================================================
# 2. Run LR LOOCV
# ============================================================

echo "Running LR LOOCV: $PYTHON_SCRIPT_LR"
echo "Saving results to: $OUTPUT_DIR_LR"
echo "Logging to: $LOG_LR"

python "$PYTHON_SCRIPT_LR" \
    --features $FEATURES_ARG \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_DIR_LR" \
    $PARTIAL_SOC_FLAG \
    > "$LOG_LR" 2>&1

echo "LR run complete."
echo "Done."
