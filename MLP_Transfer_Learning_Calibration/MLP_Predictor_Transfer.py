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
#TODO: Update empirical soc temp function -> see minimize version
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


def arrhenius_to_ref_torch(R_T: torch.Tensor,
                           T_in_C: float,
                           T_ref_C: float,
                           Ea: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Arrhenius temperature compensation:
    R(T_ref) = R(T_in) * exp( -(Ea/K)*(1/T_in - 1/T_ref) )
    Ea: tensor parameter (e.g. learnable)
    """
    T_in = T_in_C + 273.15
    T_ref = T_ref_C + 273.15
    Ea = torch.tensor(Ea)
    factor = torch.exp(-(Ea / K_BOLTZ) * (1.0 / T_in - 1.0 / T_ref))
    return R_T * factor


def empirical_soc_temp_model(R_measured, SOC, T_C, params):
    """
    Apply empirical SOC + temperature calibration to all R0-R3.
    Formula per R_i:
        R_i(SOC,T) = R_i_measured + alpha_i*SOC + beta_i*SOC² + gamma_i/T + delta_i*SOC/T
    T_C : temperature in Celsius
    """
    T_K = T_C + 273.15
    R_corr = R_measured.copy()

    for i, Ri in enumerate(["R0", "R1", "R2", "R3"]):
        alpha = params.get(f"alpha_{Ri}", 0.0)
        beta = params.get(f"beta_{Ri}", 0.0)
        gamma = params.get(f"gamma_{Ri}", 0.0)
        delta = params.get(f"delta_{Ri}", 0.0)
        R_corr[:, i] = (
            R_corr[:, i]
            + alpha * SOC                      # SOC
            + beta * (SOC ** 2)
            + gamma * (1.0 / T_K)
            + delta * (SOC / T_K)
        )

    return R_corr

def empirical_soc_temp_model_torch(R_measured: torch.Tensor,
                                   SOC: torch.Tensor,
                                   T_C: torch.Tensor,
                                   params: dict) -> torch.Tensor:
    """
    Differentiable empirical model:
        R_i(SOC, T) = R_i_measured + alpha_i*SOC + beta_i*SOC² + gamma_i/T + delta_i*SOC/T
    """
    T_K = T_C + 273.15
    R_corr = R_measured.clone()

    for i, Ri in enumerate(["R0", "R1", "R2", "R3"]):
        alpha = params.get(f"alpha_{Ri}", torch.tensor(0.0, dtype=torch.float32, device=R_measured.device))
        beta = params.get(f"beta_{Ri}", torch.tensor(0.0, dtype=torch.float32, device=R_measured.device))
        gamma = params.get(f"gamma_{Ri}", torch.tensor(0.0, dtype=torch.float32, device=R_measured.device))
        delta = params.get(f"delta_{Ri}", torch.tensor(0.0, dtype=torch.float32, device=R_measured.device))

        R_corr[:, i] = (
            R_corr[:, i]
            + alpha * SOC
            + beta * (SOC ** 2)
            + gamma * (1.0 / T_K)
            + delta * (SOC / T_K)
        )

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
        X[:, :4] = empirical_soc_temp_model(X, X[:, 4], X[:, 5], params)
    else:
        raise ValueError(f"Unknown formula: {formula}")

    # Set temperature feature to reference (after calibration, equal to 25 degree R)
    X[:, 5] = T_ref_C
    return X


def apply_calibration_torch(X_raw: torch.Tensor,
                            temp_in_C: float,
                            T_ref_C: float,
                            params: dict,
                            formula: str = "arrhenius") -> torch.Tensor:
    """
    Fully differentiable calibration function.
    X_raw columns: [R0, R1, R2, R3, SOC, Temp]
    Returns calibrated X (torch.Tensor)
    """
    X = X_raw.clone()

    if formula == "arrhenius":
        Ea_default = params.get("Ea", torch.tensor(20000.0, dtype=torch.float32, device=X.device))
        for i, key in enumerate(["Ea_R0", "Ea_R1", "Ea_R2", "Ea_R3"]):
            Ea_i = params.get(key, Ea_default)
            X[:, i] = arrhenius_to_ref_torch(X[:, i], temp_in_C, T_ref_C, Ea_i)

    elif formula == "arrhenius_plus_soc":
        # Base Arrhenius scaling
        X = apply_calibration_torch(X, temp_in_C, T_ref_C, params, "arrhenius")
        # Add SOC-dependent correction
        alpha = params.get("alpha", torch.tensor(0.0, dtype=torch.float32, device=X.device))
        beta = params.get("beta", torch.tensor(0.0, dtype=torch.float32, device=X.device))
        soc = X[:, 4]
        X[:, 1] = X[:, 1] * (1.0 + alpha * soc + beta * soc ** 2)

    elif formula == "empirical_soc":
        R_meas = X[:, 0:4]
        SOC = X[:, 4]
        T_C = X[:, 5]
        R_corr = empirical_soc_temp_model_torch(R_meas, SOC, T_C, params)
        X[:, 0:4] = R_corr

    else:
        raise ValueError(f"Unknown formula: {formula}")

    # Set temperature feature to reference
    X[:, 5] = T_ref_C
    return X


def calibrate_params_with_torch(df_temp: pd.DataFrame,
                                model: nn.Module,
                                scaler: StandardScaler,
                                device: str,
                                T_ref_C: float,
                                formula: str = "arrhenius",
                                per_resistor: bool = True,
                                test_size: float = 0.2,
                                lr: float = 1,
                                steps: int = 500) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit calibration parameters on a train split (of the given temperature dataset)
    by minimizing MSE between model predictions (after calibration) and true SOH.
    Supports: arrhenius, arrhenius_plus_soc, empirical_soc
    """
    X = df_temp[FEATURES].values
    y = df_temp[TARGET].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=123, shuffle=True)


    # ------------------------------------------------------------------
    # Define parameter sets per formula type
    # ------------------------------------------------------------------
    if formula == "arrhenius":
        if per_resistor:
            init = [30000.0, 30000.0, 30000.0, 30000.0]
            pars = torch.tensor(init, dtype=torch.float32, requires_grad=True)
            def pack_params(vec):
                return {"Ea_R0": vec[0], "Ea_R1": vec[1],
                        "Ea_R2": vec[2], "Ea_R3": vec[3]}
        else:
            pars = torch.tensor([30000.0], dtype=torch.float32, requires_grad=True)
            def pack_params(vec):
                return {"Ea": vec[0]}
        all_params = [pars]

    elif formula == "arrhenius_plus_soc":
        Ea = torch.tensor([30000.0], dtype=torch.float32, requires_grad=True)
        soc_coef = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)  # alpha, beta
        all_params = [Ea, soc_coef]

        def pack_params(vals):
            return {"Ea": Ea, "alpha": soc_coef[0], "beta": soc_coef[1]}

    elif formula == "empirical_soc":
        # Each R (R0–R3) has α, β, γ, δ → 16 parameters
        init = torch.zeros(16, dtype=torch.float32, requires_grad=True)
        all_params = [init]

        def pack_params(vec):
            out = {}
            Rs = ["R0", "R1", "R2", "R3"]
            coefs = ["alpha", "beta", "gamma", "delta"]
            k = 0
            for r in Rs:
                for c in coefs:
                    out[f"{c}_{r}"] = vec[k].item()
                    k += 1
            return out

    else:
        raise ValueError(f"Unknown formula: {formula}")

    # ------------------------------------------------------------------
    opt = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-4)

    def forward_predict(X_np, param_dict):
        X_cal = apply_calibration_torch(torch.tensor(X_np),
                                  temp_in_C=float(df_temp["Temp"].iloc[0]),
                                  T_ref_C=T_ref_C,
                                  params=param_dict,
                                  formula=formula)
        X_s = scaler.transform(X_cal)
        with torch.no_grad():
            pred = model(to_tensor(X_s).to(device)).cpu().numpy().ravel()
        return pred

    best_mse = float("inf")
    best_state = None

    # ------------------------------------------------------------------
    model.eval()
    for step in range(1, steps + 1):
       
        opt.zero_grad()
        param_dict = pack_params(all_params[0]) if formula != "arrhenius_plus_soc" \
            else pack_params([p for p in all_params])
        
        # for name, p in param_dict.items():  
        #     print(p.requires_grad)
        
        # Forward pass on train split
        X_cal_tr = apply_calibration_torch(torch.tensor(Xtr),
                                     temp_in_C=float(df_temp["Temp"].iloc[0]),
                                     T_ref_C=T_ref_C,
                                     params=param_dict,
                                     formula=formula)
        X_tr_s = scaler.transform(X_cal_tr)
        Xt = to_tensor(X_tr_s).to(device)
        yt = to_tensor(ytr).unsqueeze(1).to(device)

        
        pred = model(Xt)
        mse = ((pred - yt) ** 2).mean()

        
        mse.backward()
        opt.step()

        # Validation check
        if step % 20 == 0 or step == steps:
            with torch.no_grad():
                param_dict_eval = pack_params(all_params[0])
                yte_hat = forward_predict(Xte, param_dict_eval)
                te_mse = float(np.mean((yte_hat - yte) ** 2))
                if te_mse < best_mse:
                    best_mse = te_mse
                    best_state = [p.detach().clone() for p in all_params]

    # ------------------------------------------------------------------
    # Restore best parameters
    if best_state is not None:
        for p, best_p in zip(all_params, best_state):
            p.data = best_p

    best_params = pack_params(all_params[0])
    yte_hat = forward_predict(Xte, best_params)
    te_mse = float(np.mean((yte_hat - yte) ** 2))
    te_mae = float(np.mean(np.abs(yte_hat - yte)))

    return best_params, {"test_mse": te_mse, "test_mae": te_mae}


def evaluate_on_temperature(df_temp: pd.DataFrame,
                            model: nn.Module,
                            scaler: StandardScaler,
                            device: str,
                            T_ref_C: float,
                            outdir: str,
                            formula: str,
                            per_resistor: bool):
    # Calibrate parameters with a train/test split inside the temperature group
    best_params, metrics = calibrate_params_with_torch(
        df_temp=df_temp, model=model, scaler=scaler, device=device, T_ref_C=T_ref_C,
        formula=formula, per_resistor=per_resistor)

    # Apply best params to the entire temp group and plot
    X_np = df_temp[FEATURES].values
    y_np = df_temp[TARGET].values
    X_cal = apply_calibration(X_np, temp_in_C=float(df_temp["Temp"].iloc[0]),
                              T_ref_C=T_ref_C, params=best_params, formula=formula)
    X_s = scaler.transform(X_cal)
    with torch.no_grad():
        y_hat = model(to_tensor(X_s).to(device)).cpu().numpy().ravel()

    title = f"{int(df_temp['Temp'].iloc[0])}°C → {int(T_ref_C)}°C-equivalent ({formula})"
    scatter_true_vs_pred(y_np, y_hat, title, os.path.join(outdir, f"scatter_{int(df_temp['Temp'].iloc[0])}C.png"))

    # # Save params & metrics
    # with open(os.path.join(outdir, f"calib_params_{int(df_temp['Temp'].iloc[0])}C.json"), "w") as f:
    #     json.dump({"formula": formula, "per_resistor": per_resistor, "params": best_params, "metrics": metrics}, f, indent=2)

    return best_params, metrics

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
    print("Train 25°C base model Done.")

    # 3) Evaluate on 0°C and 45°C via calibration (each split internally for param fitting)
    for temp in [0, 45]:
        df_T = df[df["Temp"] == temp].copy()
        if len(df_T) == 0:
            print(f"[WARN] No rows for Temp={temp}°C; skipping.")
            continue
        best_params, metrics = evaluate_on_temperature(
            df_temp=df_T, model=model, scaler=scaler, device=args.device,
            T_ref_C=25.0, outdir=args.outdir, formula=args.formula, per_resistor=args.per_resistor
        )
        print(f"Temp={temp}°C best params:", best_params)
        print(f"Temp={temp}°C metrics:", metrics)

    print("All done. Outputs are in:", args.outdir)

if __name__ == "__main__":
    main()
