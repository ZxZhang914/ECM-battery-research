import os
import glob
import argparse
import pandas as pd

from common import (
    load_pickle,
    save_pickle,
    eval_payload_on_df,
    fit_ols_payload,
)

#### Example Usage ####
'''
python multisoh_pipeline.py --baseline artifacts/baseline_model.pkl --data ../fulldf_date_G40L80SOC_all.csv --cell-order "CELL090, CELL096,CELL021,CELL077,CELL070,CELL032,CELL101" --threshold 0.5 --outdir artifacts

'''



def print_scored_summary(cell_id, scored):
    print("\n--- Scored summary for cell:", cell_id, "---")
    print(f"{'model':<15} {'pct>3 sigma':>10} {'rmse':>10} {'mae':>10} {'n':>6}")
    print("-" * 55)

    for m, met in scored:
        print(
            f"{m['name']:<15} "
            f"{met['pct_over_3sigma']:>10.4f} "
            f"{met['rmse']:>10.4f} "
            f"{met['mae']:>10.4f} "
            f"{met['n']:>6d}"
        )


def main():
    parser = argparse.ArgumentParser(description="OOD multi-SOH LR pipeline.")
    parser.add_argument("--baseline", required=True, help="baseline_model.pkl path")
    # Currently, we will pass in a big df consists of all cells data
    # we will pass in cell order, simulating the cells arrive in such an order
    parser.add_argument("--data", required=True, help="Path to csv containing all cells data")
    parser.add_argument("--cell-order", default="", help="Comma-separated order of coming cells, e.g. CELL1,CELL2,CELL3")

    parser.add_argument("--soc-min", type=float, default=0.4,
                    help="Minimum SOC (inclusive), float number")
    parser.add_argument("--soc-max", type=float, default=0.8,
                    help="Maximum SOC (inclusive), float number")
    
    parser.add_argument("--threshold", type=float, default=0.1,  help="OOD threshold t (default=0.1)")
    parser.add_argument("--outdir", default="artifacts", help="Output directory (default=artifacts)")
    parser.add_argument("--record-name", default="pipeline_record.csv", help="Record CSV filename")
    parser.add_argument("--registry-name", default="model_registry.pkl", help="Registry PKL filename")
    args = parser.parse_args()

   
    
    # Load baseline model
    baseline = load_pickle(args.baseline)
    baseline["name"] = baseline.get("name", "baseline")

    # Load Incoming cell data (all cells)
    df = pd.read_csv(args.data)
    df = df[(df["SOC"] >= args.soc_min) & (df["SOC"] <= args.soc_max)]
    if "CELL" not in df.columns:
        raise ValueError("'CELL' not found in data columns.")
    
    df["CELL"] = df["CELL"].astype(str)

    # Parse cell order
    order = [x.strip() for x in args.cell_order.split(",") if x.strip()]
    if order is None: # Note by default include all cells in the df.
        order = sorted(df["CELL"].unique().tolist())
        print("No explicit order provided; using sorted unique cell ids from the file.")
    # validate order
    avaliable = set(df["CELL"].unique().tolist())
    missing = [cid for cid in order if str(cid) not in avaliable]
    if missing:
        raise ValueError(f"These cell ids in the provided order are missing from the data: {missing[:20]}")
    print("Cell processing order:", order)
    
    # Build cell -> dataframe map
    grouped = dict(tuple(df.groupby("CELL", sort=False)))

    registry = {
        "models": [baseline],
        "next_model_id": 1,
    }

    records = []

    for i, cell_id in enumerate(order, start=1):
        cell_id = str(cell_id)
        cell_df = grouped[cell_id].copy()
        print(f"\nProcessing cell {cell_id} with {len(cell_df)} rows...")

        # Evaluate (OOD) against all existing models
        scored = []
        for m in registry["models"]:
            met = eval_payload_on_df(m, cell_df)
            if met["n"] > 0:
                scored.append((m, met))
        
        if not scored:
            records.append({
                "cell_id": cell_id,
                "decision": "skipped_no_valid_rows",
                "chosen_model": None,
                "threshold_t": args.threshold,
                "num_models_total": len(registry["models"]),
            })
            continue
        
        print_scored_summary(cell_id, scored)

        # Choose best: lowest pct_over_3sigma, tie-break rmse
        scored.sort(key=lambda x: (x[1]["pct_over_3sigma"], x[1]["rmse"]))
        best_model, best_met = scored[0]

        # OOD Threshold
        is_ood = best_met["pct_over_3sigma"] > args.threshold
        print("This cell is considered", "OOD" if is_ood else "InD",)

        if is_ood: # build new model
            new_name = f"model_{registry['next_model_id']:03d}"
            new_payload = fit_ols_payload(
                cell_df,
                best_model["target_col"],
                best_model["feature_cols"],
                model_name=new_name,
            )
            registry["models"].append(new_payload)
            registry["next_model_id"] += 1

            chosen_model = new_name
            decision = "ood_new_model_fitted"
            chosen_met = eval_payload_on_df(new_payload, cell_df)
        else:
            chosen_model = best_model["name"]
            decision = "in_dist_use_existing"
            chosen_met = best_met

        records.append({
            "cell_id": cell_id,
            "decision": decision,
            "chosen_model": chosen_model,
            "best_model_before_decision": best_model["name"],
            "best_pct_over_3sigma_before_decision": best_met["pct_over_3sigma"],
            "best_rmse_before_decision": best_met["rmse"],
            "chosen_pct_over_3sigma": chosen_met["pct_over_3sigma"],
            "chosen_rmse": chosen_met["rmse"],
            "chosen_mae": chosen_met["mae"],
            "n": chosen_met["n"],
            "threshold_t": args.threshold,
            "num_models_total": len(registry["models"]),
        })

        print(f"[{i}/{len(order)}] cell={cell_id} -> {decision} ({chosen_model})")
    
    
    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    record_path = os.path.join(args.outdir, args.record_name)
    registry_path = os.path.join(args.outdir, args.registry_name)

    pd.DataFrame(records).to_csv(record_path, index=False)
    save_pickle(registry, registry_path)

    print("\nPipeline finished.")
    print(f"Record saved to:   {record_path}")
    print(f"Registry saved to: {registry_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error during multisoh_pipeline execution:")
        print(e)
        raise e