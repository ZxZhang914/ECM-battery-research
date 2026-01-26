"""
MLP on ECM features (R0,R1,R2,R3,SOC,Temp) to predict SOH,
trained only on 25°C data. For other temperatures (0°C, 45°C),
we calibrate the resistance features R0-R3 to 25°C equivalents
using physics-inspired formulas (Arrhenius, with an optional
SOC-aware term for R1).

Usage:
    python MLP_Predictor_Transfer.py --file_path ../df_global_all.csv --outdir ./outputs --formula arrhenius --epochs 200 --per_resistor

CSV expectations (columns):
    CELL, SOH, SOC, Temp, R0, R1, R2, R3
"""
import argparse
import os
import math
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.switch_backend("Agg")  # for non-interactive environments

# ----------------------------
# 1) Model
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64], p_drop: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p_drop)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Utility
# ----------------------------
K_BOLTZ = 8.314  # J/(mol*K), universal gas constant

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

def scatter_true_vs_pred(y_true, y_pred, title: str, outpath: str):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ----------------------------
# Data handling
# ----------------------------
FEATURES = ["R0","R1","R2","R3","SOC","Temp"]
TARGET = "SOH"
REQ_COLS = ["CELL","SOH","SOC","Temp","R0","R1","R2","R3"]

def load_and_clean(file_path: str) -> pd.DataFrame:
    df_all = pd.read_csv(file_path, index_col=0)
    for col in REQ_COLS:
        if col not in df_all.columns:
            raise ValueError(f"All-trials df missing column: {col}")
    df_all = df_all.dropna(subset=REQ_COLS).reset_index(drop=True)
    return df_all

def split_25C(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df_25 = df[df["Temp"] == 25].copy()
    X = df_25[FEATURES].values
    y = df_25[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return df_25, Xtr, Xte, ytr, yte

# ----------------------------
# Training
# ----------------------------

def evaluate(loader, model, device) -> Tuple[float, float, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            preds.append(out.squeeze(1).cpu().numpy())
            trues.append(yb.squeeze(1).cpu().numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0
    return mae, rmse, mape


def train_mlp_25C(Xtr, ytr, Xte, yte, device: str, outdir: str,
                  hidden=[128,64,32], p_drop=0.1, lr=1e-3, epochs=200, batch_size=128):
    scaler = StandardScaler()
    scaler.fit(Xtr)                  # fit on 25°C TRAIN ONLY
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = MLPRegressor(in_dim=Xtr.shape[1], hidden=hidden, p_drop=p_drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    ds_tr = TensorDataset(to_tensor(Xtr_s), to_tensor(ytr).unsqueeze(1))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    best_te = float("inf")
    best_state=None
    best_epoch = 0

    print(f"\n Training MLP on 25°C data...")
    print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Test MSE':>12} | {'Test MAE':>12} | {'R²':>8}")
    for ep in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        train_mse = epoch_loss / len(Xtr_s)

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                yte_hat = model(to_tensor(Xte_s).to(device)).cpu().numpy().ravel()

            te_mse = np.mean((yte_hat - yte) ** 2)
            te_mae = np.mean(np.abs(yte_hat - yte))
            ss_res = np.sum((yte - yte_hat) ** 2)
            ss_tot = np.sum((yte - np.mean(yte)) ** 2)
            r2 = 1 - ss_res / ss_tot

            print(f"{ep:6d} | {train_mse:12.6f} | {te_mse:12.6f} | {te_mae:12.6f} | {r2:8.4f}")

            if te_mse < best_te:
                best_te = te_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = ep

    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot 25°C true vs. predicted
    model.eval()
    with torch.no_grad():
        yte_hat = model(to_tensor(Xte_s).to(device)).cpu().numpy().ravel()
    scatter_true_vs_pred(yte, yte_hat, "25°C: True vs Predicted SOH",
                         os.path.join(outdir, "scatter_25C.png"))
    te_mse = np.mean((yte_hat - yte) ** 2)
    te_mae = np.mean(np.abs(yte_hat - yte))
    ss_res = np.sum((yte - yte_hat) ** 2)
    ss_tot = np.sum((yte - np.mean(yte)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"25°C Test-set size:{len(yte)}")
    print(f"25°C Test-set metrics: {ep:6d} | {te_mse:12.6f} | {te_mae:12.6f} | {r2:8.4f}")

    # Persist
    torch.save(model.state_dict(), os.path.join(outdir, "mlp_25C.pt"))
    with open(os.path.join(outdir, "scaler_25C.json"), "w") as f:
        json.dump({"mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist(),
                   "feat_order": FEATURES}, f)
    print(f"Training complete. Best test MSE = {best_te:.6f} (epoch {best_epoch})")
    return model, scaler

# ----------------------------
# Temperature calibration
# ----------------------------
def arrhenius_to_ref(R_T: np.ndarray, T_in_C: float, T_ref_C: float, Ea: float) -> np.ndarray:
    """
    Convert resistance measured at T_in to equivalent at T_ref using
    Arrhenius relative form:
        R(T_in)/R(T_ref) = exp( (Ea/K)*(1/T_in - 1/T_ref) )
    ->  R(T_ref) = R(T_in) * exp( -(Ea/K)*(1/T_in - 1/T_ref) )
    Temperatures in Kelvin inside function; Ea in J/mol.
    """
    T_in = T_in_C + 273.15
    T_ref = T_ref_C + 273.15
    factor = math.exp( -(Ea / K_BOLTZ) * (1.0/T_in - 1.0/T_ref) )
    return R_T * factor


def empirical_soc_temp_model(R_measured, SOC, T_C, params, T_ref_C=25.0):
    """
    Corrected empirical SOC + temperature calibration model.

    R(SOC; T2) = R(SOC; T1)
               + (γ + δ*SOC) * (1/T2 - 1/T1)

    T1: reference temperature (e.g., 25°C)
    T2: measured temperature (input)
    """
    T1_K = T_ref_C + 273.15
    T2_K = T_C + 273.15
    dT_term = (1.0 / T2_K - 1.0 / T1_K)

    R_corr = R_measured.copy()
    for i, Ri in enumerate(["R0", "R1", "R2", "R3"]):
        gamma = params.get(f"gamma_{Ri}", 0.0)
        delta = params.get(f"delta_{Ri}", 0.0)
        R_corr[:, i] = R_corr[:, i] + (gamma + delta * SOC) * dT_term

    return R_corr


def apply_calibration(X_raw: np.ndarray,
                      temp_in_C: float,
                      T_ref_C: float,
                      params: Dict[str, float],
                      formula: str = "arrhenius") -> np.ndarray:
    """
    X_raw columns follow FEATURES: [R0,R1,R2,R3,SOC,Temp]
    Returns calibrated version where R0-R3 are mapped to reference temperature
    and Temp feature is set to T_ref_C (so the 25°C-trained model sees a familiar domain).
    """
    X = X_raw.copy()
    if formula == "arrhenius":
        Ea = params.get("Ea", 20000.0)  # default 20 kJ/mol
        # Optionally different Ea per resistor
        for i, key in enumerate(["Ea_R0","Ea_R1","Ea_R2","Ea_R3"]):
            Ea_i = params.get(key, Ea)
            X[:, i] = arrhenius_to_ref(X[:, i], temp_in_C, T_ref_C, Ea_i)
    elif formula == "arrhenius_plus_soc":
        # Arrhenius scaling first
        X = apply_calibration(X, temp_in_C, T_ref_C, params, "arrhenius")
        # Then a simple SOC-aware correction on R1: #NOTE: This is a slightly add-on version of Arrhenius function (#TODO: NEED TO UPDATE)
        a = params.get("alpha", 0.0)
        b = params.get("beta", 0.0)
        # R1*(1 + a*SOC + b*SOC^2) as a smooth correction
        X[:, 1] = X[:, 1] * (1.0 + a*X[:, 4] + b*(X[:, 4]**2))
    # In calibration function:
    elif formula == "empirical_soc":
        X = empirical_soc_temp_model(X, X[:, 4], X[:, 5], params)
    else:
        raise ValueError(f"Unknown formula: {formula}")

    # Set temperature feature to reference (after calibration, equal to 25 degree R)
    X[:, 5] = T_ref_C
    return X

def calibrate_params_with_scipy(df_temp: pd.DataFrame,
                                model: nn.Module,
                                scaler: StandardScaler,
                                device: str,
                                T_ref_C: float,
                                formula: str = "arrhenius",
                                per_resistor: bool = True,
                                max_iter: int = 500) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit calibration parameters using SciPy optimization (L-BFGS-B)
    to minimize MSE between predicted and true SOH.
    Supports formulas: arrhenius, arrhenius_plus_soc, empirical_soc
    """
    
    X = df_temp[FEATURES].values
    y = df_temp[TARGET].values

    # ---------------------------------------------------------
    # Initialize parameters and bounds depending on formula
    # ---------------------------------------------------------
    if formula == "arrhenius":
        if per_resistor:
            init = np.full(4, 30000.0)
            names = ["Ea_R0", "Ea_R1", "Ea_R2", "Ea_R3"]
        else:
            init = np.array([30000.0])
            names = ["Ea"]
        bounds = [(10000, 80000)] * len(init)

    elif formula == "arrhenius_plus_soc":
        init = np.array([30000.0, 0.0, 0.0])  # Ea, alpha, beta
        names = ["Ea", "alpha", "beta"]
        bounds = [(10000, 80000), (-2.0, 2.0), (-2.0, 2.0)]

    elif formula == "empirical_soc":
        names, init, bounds = [], [], []
        for Ri in ["R0", "R1", "R2", "R3"]:
            for c in ["gamma", "delta"]:
                names.append(f"{c}_{Ri}")
                init.append(0.0)
                bounds.append((-5.0, 5.0))
        init = np.array(init)

    else:
        raise ValueError(f"Unknown formula: {formula}")

    # ---------------------------------------------------------
    # Objective function (MSE)
    # ---------------------------------------------------------
    def unpack_params(vec):
        return {name: val for name, val in zip(names, vec)}

    def objective(vec):
        params = unpack_params(vec)
        X_cal = apply_calibration(X.copy(),
                                  temp_in_C=float(df_temp["Temp"].iloc[0]),
                                  T_ref_C=T_ref_C,
                                  params=params,
                                  formula=formula)
        X_s = scaler.transform(X_cal)
        with torch.no_grad():
            pred = model(to_tensor(X_s).to(device)).cpu().numpy().ravel()
        return np.mean((pred - y)**2)

    # ---------------------------------------------------------
    # Run SciPy minimization
    # ---------------------------------------------------------
    res = minimize(objective, init, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": max_iter, "disp": True})

    best_params = unpack_params(res.x)

    # ---------------------------------------------------------
    # Compute metrics on calibration set
    # ---------------------------------------------------------
    X_cal_best = apply_calibration(X.copy(),
                                   temp_in_C=float(df_temp["Temp"].iloc[0]),
                                   T_ref_C=T_ref_C,
                                   params=best_params,
                                   formula=formula)
    X_s_best = scaler.transform(X_cal_best)
    with torch.no_grad():
        y_pred = model(to_tensor(X_s_best).to(device)).cpu().numpy().ravel()

    mse = np.mean((y_pred - y) ** 2)
    mae = np.mean(np.abs(y_pred - y))
    mape = np.mean(np.abs((y_pred - y) / np.clip(np.abs(y), 1e-6, None))) * 100.0
    r2 = r2_score(y, y_pred)

    metrics_calib = {
        "mse": float(mse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "success": bool(res.success),
        "n_iter": int(res.nit)
    }

    return best_params, metrics_calib

def evaluate_on_temperature(df_temp: pd.DataFrame,
                            model: nn.Module,
                            scaler: StandardScaler,
                            device: str,
                            test_size: float, # this is train_test_split param
                            calibration_size: int,
                            T_ref_C: float,
                            outdir: str,
                            formula: str,
                            per_resistor: bool):
    # Calibrate parameters with a train/test split inside the temperature group
    
    X = df_temp[FEATURES].values
    y = df_temp[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=123, shuffle=True)

    # Take only a limited subset from training data for calibration
    n_calib = min(calibration_size, len(Xtr))
    calib_idx = np.random.choice(len(Xtr), n_calib, replace=False)
    Xcal, ycal = Xtr[calib_idx], ytr[calib_idx]

    df_calib = pd.DataFrame(Xcal, columns=FEATURES)
    df_calib["SOH"] = ycal
    df_calib["Temp"] = df_temp["Temp"].iloc[0] 

    best_params, metrics_calib = calibrate_params_with_scipy(
        df_temp=df_calib, model=model, scaler=scaler, device=device, T_ref_C=T_ref_C,
        formula=formula, per_resistor=per_resistor)

    # Apply best params to the test group and plot
    X_cal_te = apply_calibration(Xte,
                                 temp_in_C=float(df_temp["Temp"].iloc[0]),
                                 T_ref_C=T_ref_C,
                                 params=best_params,
                                 formula=formula)
    X_te_s = scaler.transform(X_cal_te)
    with torch.no_grad():
        y_hat = model(to_tensor(X_te_s).to(device)).cpu().numpy().ravel()
    test_mse = np.mean((y_hat - yte) ** 2)
    test_mae = np.mean(np.abs(y_hat - yte))
    test_mape = np.mean(np.abs((y_hat - yte) / np.clip(np.abs(yte), 1e-6, None))) * 100.0
    test_r2  = r2_score(yte, y_hat)

    metrics_test = {
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_mape": float(test_mape),
        "test_r2":  float(test_r2)
    }

    title = f"{int(df_temp['Temp'].iloc[0])}°C → {int(T_ref_C)}°C-equivalent ({formula})"
    scatter_true_vs_pred(yte, y_hat, title,
                         os.path.join(outdir, f"scatter_{int(df_temp['Temp'].iloc[0])}C.png"))

    print(f"Temperature = {int(df_temp['Temp'].iloc[0])}°C")
    print(f"  Calibration subset size = {n_calib}")
    print(f"  Best calibration params = {best_params}")
    print(f"  Calibration (train-subset) metrics: {metrics_calib}")
    print(f"  Test size: {len(yte)}")
    print(f"  Test-set metrics: {metrics_test}\n")

    return best_params, {"calib_metrics": metrics_calib, "test_metrics": metrics_test}
# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--formula", type=str, choices=["arrhenius","arrhenius_plus_soc", "empirical_soc"], default="arrhenius")
    parser.add_argument("--per_resistor", action="store_true", help="learn separate Ea for R0..R3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load & clean
    df = load_and_clean(args.file_path)
    print("Load Dataframe Done.")

    # 2) Train 25°C model (random split within 25°C group)
    df_25, Xtr, Xte, ytr, yte = split_25C(df)
    model, scaler = train_mlp_25C(
        Xtr, ytr, Xte, yte,
        device=args.device,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # Load from saved pt [#TODO:inconsistent]
    # in_dim = 6  # ["R0","R1","R2","R3","SOC","Temp"]
    # hidden_layers = [128, 64, 32]
    # p_drop = 0.1

    # model = MLPRegressor(in_dim=in_dim, hidden=hidden_layers, p_drop=p_drop)
    # model.load_state_dict(torch.load(os.path.join(args.outdir, "mlp_25C.pt"), map_location="cpu"))
    # with open(os.path.join(args.outdir, "scaler_25C.json"), "r") as f:
    #     sdata = json.load(f)
    # scaler = StandardScaler()
    # scaler.mean_ = np.array(sdata["mean"])
    # scaler.scale_ = np.array(sdata["scale"])            
    
    print("Train 25°C base model Done.")

    # 3) Evaluate on 0°C and 45°C via calibration (each split internally for param fitting)
    for temp in [0, 45]:
        df_T = df[df["Temp"] == temp].copy()
        if len(df_T) == 0:
            print(f"[WARN] No rows for Temp={temp}°C; skipping.")
            continue
        _, _ = evaluate_on_temperature(
            df_temp=df_T, model=model, scaler=scaler, device=args.device, test_size=0.2, calibration_size=5000,
            T_ref_C=25.0, outdir=args.outdir, formula=args.formula, per_resistor=args.per_resistor
        )
       

    print("All done. Outputs are in:", args.outdir)

if __name__ == "__main__":
    main()
