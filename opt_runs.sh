#!/bin/bash

MAX_JOBS=5
# CELLS=("CELL009" "CELL021" "CELL032" "CELL042" "CELL050" "CELL070" "CELL077" "CELL090" "CELL101")
# CELLS=("CELL013" "CELL045" "CELL054" "CELL076" "CELL096")
# CELLS=("CELL009" "CELL021" "CELL032" "CELL042" "CELL050" "CELL070" "CELL077" "CELL090" "CELL101" "CELL013" "CELL045" "CELL054" "CELL076" "CELL096")
CELLS=("CELL042")
ECM_NAME="v3CM10"


for CELL in "${CELLS[@]}"; do
  python opt_driver.py --cell_name "$CELL" --ECM_name "$ECM_NAME" --num_trials 100 &
  
  # If too many jobs, wait for one to finish
  while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
done

wait
echo "All done!"