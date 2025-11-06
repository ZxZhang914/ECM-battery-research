import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Reproducibility & device
# ----------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# Dataset and Model Definitions
# ================================================================
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


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

# model = MLPRegressor(in_dim=len(FEATURES), hidden=[128, 64, 32], p_drop=0.1).to(DEVICE)

# ================================================================
# Train, Evaluation Functions
# ================================================================
def evaluate(loader, net) -> Tuple[float, float, float]:
    net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            out = net(xb)
            preds.append(out.squeeze(1).cpu().numpy())
            trues.append(yb.squeeze(1).cpu().numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0
    return mae, rmse, mape


class EarlyStopper:
    def __init__(self, patience=30, min_delta=1e-5):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best = np.inf; self.state = None
    def step(self, metric, model):
        if (self.best - metric) > self.min_delta:
            self.best = metric; self.counter = 0
            self.state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter < self.patience
    def load_best(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)


def train(model, train_loader, val_loader=None, epochs=400):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    early = EarlyStopper(patience=30, min_delta=1e-5)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)

        # ✅ Validation step (optional)
        if val_loader is not None:
            val_mae, val_rmse, val_mape = evaluate(val_loader, model)
            scheduler.step(val_mae)

            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:03d} | train_RMSE: {math.sqrt(train_loss):.4f} | "
                      f"val_MAE: {val_mae:.4f} | val_RMSE: {val_rmse:.4f} | val_MAPE: {val_mape:.2f}%")

            if not early.step(val_mae, model):
                print(f"Early stop at epoch {ep}. Best val MAE: {early.best:.4f}")
                break
        else:
            # ✅ No validation — just print training loss and step scheduler on it
            scheduler.step(train_loss)
            if ep % 10 == 0 or ep == 1:
                print(f"Epoch {ep:03d} | train_RMSE: {math.sqrt(train_loss):.4f}")

    # ✅ Load best weights if early stopping was used (and val_loader existed)
    if val_loader is not None:
        early.load_best(model)

    return model