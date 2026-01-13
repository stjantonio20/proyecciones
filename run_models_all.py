#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== determinismo / CPU ======
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
random.seed(24)
np.random.seed(24)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
# ======= EXPORT GLOBAL (ancho) =======
EXPORT_WIDE_FUTURE = True
WIDE_PREFIX = "wide_future"  # nombre base de archivos
# ====================================

CSV_PATH = "Crediguate_formato_mensual.csv"
OUT_DIR  = "outputs_modelos_extra"

H_FUTURE = 12          # 1 año forecast
TEST_LEN = 6           # 6 meses test fijo

#ONLY_CODIGO = "101101" # o None para todos
ONLY_CODIGO = None # o None para todos

LOOKBACK_NN  = 16      # para LSTM/TCN
LAGS_TABULAR = 28      # para Linear/Ridge/MLP/HGB (roll lags)

# switches
RUN_ETS          = True
RUN_TCN          = True
RUN_LSTM         = True
RUN_MULTITASK_DL = True

RUN_LINEAR       = True
RUN_RIDGE        = True
RUN_MLP          = True
RUN_SARIMAX      = True
RUN_HGB          = True   # HGB (log|p+roll lags=28)

# NN params
NN_EPOCHS   = 500
NN_BATCH    = 20
NN_PATIENCE = 10


# =========================================================
# Utils: parse columnas tipo "Mar-15"
# =========================================================
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def parse_month_col(col: str) -> pd.Timestamp:
    col = str(col).strip()
    mon, yy = col.split("-")
    mon = mon[:3]
    year = 2000 + int(yy)
    month = MONTH_MAP[mon]
    return pd.Timestamp(year=year, month=month, day=1)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def is_all_zero_series(s: pd.Series) -> bool:
    v = pd.to_numeric(s.values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return True
    return np.nanmax(np.abs(v)) == 0.0

def fmt_month_cols(idx: pd.DatetimeIndex) -> List[str]:
    return [d.strftime("%b-%y") for d in pd.DatetimeIndex(idx)]

def export_wide_future_csv(out_path: str, rows: Dict[str, Dict[str, float]], month_cols: List[str]):
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "codigo"
    for c in month_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[month_cols]
    df.reset_index().to_csv(out_path, index=False)


# =========================================================
# Scaling robusto (asinh + Standard) para NNs
# =========================================================
@dataclass
class AsinhScaler:
    s: float
    mu: float
    sigma: float

def fit_asinh_scaler(y: np.ndarray) -> AsinhScaler:
    y = np.asarray(y, float)
    s = np.nanmedian(np.abs(y))
    if not np.isfinite(s) or s == 0:
        s = 1.0
    z = np.arcsinh(y / s)
    mu = float(np.nanmean(z))
    sigma = float(np.nanstd(z))
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0
    return AsinhScaler(float(s), mu, sigma)

def transform_asinh(y: np.ndarray, sc: AsinhScaler) -> np.ndarray:
    y = np.asarray(y, float)
    z = np.arcsinh(y / sc.s)
    return (z - sc.mu) / sc.sigma

def inverse_asinh(z_scaled: np.ndarray, sc: AsinhScaler) -> np.ndarray:
    z_scaled = np.asarray(z_scaled, float)
    z = z_scaled * sc.sigma + sc.mu
    y = np.sinh(z) * sc.s
    return y


# =========================================================
# Windows supervisadas (NNs)
# =========================================================
def make_supervised(y_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    y_scaled = np.asarray(y_scaled, float).reshape(-1)
    X, Y = [], []
    for i in range(lookback, len(y_scaled)):
        X.append(y_scaled[i - lookback:i].reshape(lookback, 1))
        Y.append([y_scaled[i]])
    return np.array(X, float), np.array(Y, float)

def month_features(idx: pd.DatetimeIndex) -> np.ndarray:
    m = idx.month.values.astype(float)
    sinm = np.sin(2*np.pi*m/12.0)
    cosm = np.cos(2*np.pi*m/12.0)
    return np.c_[sinm, cosm]


# =========================================================
# (ETS) Holt-Winters
# =========================================================
def fit_predict_ets(y_train: pd.Series, test_len: int, h_future: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = "add" if len(y_train) >= 36 else None
    trend = "add" if len(y_train) >= 24 else None

    model = ExponentialSmoothing(
        y_train.values,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)

    yhat_test = model.forecast(test_len)
    yhat_fut  = model.forecast(test_len + h_future)[-h_future:]
    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# Tabular features: lags + estacionalidad
# =========================================================
def build_tabular_XY(y: pd.Series, lags: int, use_log1p: bool) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    yv = y.values.astype(float)
    idx = y.index

    if use_log1p:
        y_t = np.log1p(np.maximum(yv, 0.0))
    else:
        y_t = yv

    X, Y, X_idx = [], [], []
    for t in range(lags, len(y_t)):
        l = y_t[t-lags:t]
        feats = np.r_[l, month_features(pd.DatetimeIndex([idx[t]])).ravel()]
        X.append(feats)
        Y.append(y_t[t])
        X_idx.append(idx[t])
    return np.array(X, float), np.array(Y, float), pd.DatetimeIndex(X_idx)

def tabular_forecast_autoreg(model, y_full: pd.Series, lags: int, h_future: int, use_log1p: bool) -> np.ndarray:
    yv = y_full.values.astype(float)
    idx_last = y_full.index[-1]
    fut_idx = pd.date_range(start=idx_last + pd.offsets.MonthBegin(1), periods=h_future, freq="MS")

    if use_log1p:
        buf = np.log1p(np.maximum(yv, 0.0)).tolist()
    else:
        buf = yv.tolist()

    preds_t = []
    for ts in fut_idx:
        l = np.array(buf[-lags:], float)
        feats = np.r_[l, month_features(pd.DatetimeIndex([ts])).ravel()].reshape(1, -1)
        y_next_t = float(model.predict(feats)[0])
        preds_t.append(y_next_t)
        buf.append(y_next_t)

    preds_t = np.asarray(preds_t, float)
    if use_log1p:
        return np.expm1(preds_t)
    return preds_t


# =========================================================
# Tabular models: Linear / Ridge / MLP / HGB(log1p)
# =========================================================
def fit_predict_tabular_model(
    y: pd.Series,
    train_end_i: int,
    test_len: int,
    h_future: int,
    lags: int,
    model_kind: str,
    use_log1p: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, Y, X_idx = build_tabular_XY(y, lags=lags, use_log1p=use_log1p)

    cutoff = y.index[train_end_i]
    train_mask = (X_idx < cutoff)
    test_mask  = (X_idx >= cutoff)

    Xtr, Ytr = X[train_mask], Y[train_mask]
    Xte, Yte = X[test_mask],  Y[test_mask]
    Xte, Yte = Xte[:test_len], Yte[:test_len]

    if len(Xtr) < 20:
        raise ValueError(f"Tabular {model_kind}: muy pocos datos train (<20)")

    if model_kind == "Linear":
        base = LinearRegression()
        model = Pipeline([("sc", StandardScaler()), ("m", base)])
    elif model_kind == "Ridge":
        base = Ridge(alpha=1.0, random_state=42)
        model = Pipeline([("sc", StandardScaler()), ("m", base)])
    elif model_kind == "MLP":
        base = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            random_state=42
        )
        model = Pipeline([("sc", StandardScaler()), ("m", base)])
    elif model_kind == "HGB":
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=42
        )
    else:
        raise ValueError("model_kind no soportado")

    model.fit(Xtr, Ytr)

    yhat_test_t = model.predict(Xte)
    if use_log1p:
        yhat_test = np.expm1(yhat_test_t)
    else:
        yhat_test = yhat_test_t

    yhat_fut = tabular_forecast_autoreg(model, y, lags=lags, h_future=h_future, use_log1p=use_log1p)
    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# SARIMAX
# =========================================================
def fit_predict_sarimax(y_train: pd.Series, test_len: int, h_future: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12) if len(y_train) >= 36 else (0, 0, 0, 0)

    mod = SARIMAX(
        y_train.values,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    yhat_test = mod.forecast(steps=test_len)
    yhat_fut  = mod.forecast(steps=test_len + h_future)[-h_future:]
    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# Keras builders: TCN / LSTM / MultiTask DL
# =========================================================
def get_tf():
    import tensorflow as tf
    try:
        tf.keras.utils.set_random_seed(42)
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass
    return tf

def build_lstm(input_len: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_len, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
    return m

def build_tcn(input_len: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    try:
        from tcn import TCN
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
        from tcn import TCN

    inp = layers.Input(shape=(input_len, 1))
    x = TCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
        padding="causal",
        dropout_rate=0.2,
        return_sequences=False,
        use_skip_connections=True,
    )(inp)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
    return m

def build_multitask_dl(input_len: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    y_level = layers.Dense(1, name="y_level")(x)
    y_delta = layers.Dense(1, name="y_delta")(x)
    m = models.Model(inp, [y_level, y_delta])
    m.compile(
        optimizer="adam",
        loss={"y_level": "mae", "y_delta": "mae"},
        loss_weights={"y_level": 0.8, "y_delta": 0.2},
    )
    return m

def train_keras_model(model, Xtr, Ytr, Xva, Yva, multitask=False):
    from tensorflow.keras.callbacks import EarlyStopping
    cb = [EarlyStopping(monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True)]

    if not multitask:
        model.fit(
            Xtr, Ytr,
            validation_data=(Xva, Yva),
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH,
            shuffle=False,
            verbose=0,
            callbacks=cb,
        )
    else:
        ytr_level = Ytr
        ytr_delta = (Ytr.reshape(-1) - Xtr[:, -1, 0]).reshape(-1, 1)
        yva_level = Yva
        yva_delta = (Yva.reshape(-1) - Xva[:, -1, 0]).reshape(-1, 1)

        model.fit(
            Xtr, {"y_level": ytr_level, "y_delta": ytr_delta},
            validation_data=(Xva, {"y_level": yva_level, "y_delta": yva_delta}),
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH,
            shuffle=False,
            verbose=0,
            callbacks=cb,
        )

def predict_keras_one_step(model, X):
    yhat = model.predict(X, verbose=0)
    if isinstance(yhat, list):
        return np.asarray(yhat[0]).reshape(-1)
    return np.asarray(yhat).reshape(-1)

def forecast_keras_autoregressive(model, last_window_scaled: np.ndarray, h_future: int):
    buf = last_window_scaled.astype(float).reshape(-1).tolist()
    preds = []
    L = len(last_window_scaled)
    for _ in range(h_future):
        x = np.array(buf[-L:], float).reshape(1, L, 1)
        yhat = model.predict(x, verbose=0)
        if isinstance(yhat, list):
            y_next = float(yhat[0].reshape(-1)[0])
        else:
            y_next = float(np.asarray(yhat).reshape(-1)[0])
        preds.append(y_next)
        buf.append(y_next)
    return np.array(preds, float)


# =========================================================
# Métricas / plots / export
# =========================================================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return float("nan")
    e = y_true[:m] - y_pred[:m]
    return float(np.sqrt(np.mean(e*e)))

def plot_all_models(
    codigo: str,
    y: pd.Series,
    train_end: pd.Timestamp,
    test_idx: pd.DatetimeIndex,
    future_idx: pd.DatetimeIndex,
    preds_test: Dict[str, np.ndarray],
    preds_future: Dict[str, np.ndarray],
    out_png: str
):
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Real", linewidth=2)
    plt.axvline(train_end, linestyle="--", linewidth=1)
    plt.text(train_end, np.nanmin(y.values), "  train_end", rotation=90, va="bottom")

    for name, p in preds_test.items():
        if p is None:
            continue
        m = min(len(test_idx), len(p))
        plt.plot(test_idx[:m], p[:m], label=f"{name} (test)")

    for name, p in preds_future.items():
        if p is None:
            continue
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (future)", linestyle=":")

    plt.title(f"Codigo {codigo} — Test {len(test_idx)}m + Forecast {len(future_idx)}m")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_forecast_only(
    codigo: str,
    y: pd.Series,
    train_end: pd.Timestamp,
    future_idx: pd.DatetimeIndex,
    preds_future: Dict[str, np.ndarray],
    out_png: str
):
    """
    Gráfica SOLO forecast (sin test):
      - muestra TODA la serie real completa + forecast de todos los modelos
      - NO dibuja líneas de pred_test
    """
    plt.figure(figsize=(13, 5))

    # serie completa
    plt.plot(y.index, y.values, label="Real", linewidth=2)

    # línea de corte train/test (solo referencia visual)
    if train_end is not None:
        plt.axvline(train_end, linestyle="--", linewidth=1)
        # coloca el texto cerca del mínimo de la serie (con fallback)
        y_min = np.nanmin(y.values) if np.isfinite(np.nanmin(y.values)) else 0.0
        plt.text(train_end, y_min, "  train_end", rotation=90, va="bottom")

    # solo forecasts
    for name, p in preds_future.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (forecast)", linestyle="-")

    plt.title(f"Codigo {codigo} — Forecast {len(future_idx)}m (sin test)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def export_csv_codigo(
    out_csv: str,
    y: pd.Series,
    test_idx: pd.DatetimeIndex,
    future_idx: pd.DatetimeIndex,
    preds_test: Dict[str, np.ndarray],
    preds_future: Dict[str, np.ndarray]
):
    df = pd.DataFrame({"fecha": y.index, "real": y.values}).set_index("fecha")

    for name, p in preds_test.items():
        if p is None:
            continue
        df[f"pred_test_{name}"] = pd.Series(p, index=test_idx[:len(p)])

    for name, p in preds_future.items():
        if p is None:
            continue
        df[f"pred_future_{name}"] = pd.Series(p, index=future_idx[:len(p)])

    df.reset_index().to_csv(out_csv, index=False)


# =========================================================
# Read dataset ancho -> serie mensual por codigo
# =========================================================
def read_wide_monthly(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="|", engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    for c in df.columns:
        if c.lower() == "codigo" and c != "codigo":
            df = df.rename(columns={c: "codigo"})
            break
    if "codigo" not in df.columns:
        raise ValueError("No existe columna 'codigo' en el CSV.")

    df["codigo"] = df["codigo"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    month_cols = [c for c in df.columns if c != "codigo"]
    col_to_ts = {c: parse_month_col(c) for c in month_cols}

    long = df.melt(id_vars=["codigo"], value_vars=month_cols, var_name="mes", value_name="valor")
    long["fecha"] = long["mes"].map(col_to_ts)
    long["valor"] = pd.to_numeric(long["valor"], errors="coerce")
    long = long.dropna(subset=["fecha"]).sort_values(["codigo", "fecha"])
    return long

def series_by_codigo(long: pd.DataFrame, codigo: str) -> pd.Series:
    g = long[long["codigo"] == codigo].copy()
    s = pd.Series(g["valor"].values, index=pd.to_datetime(g["fecha"]))
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp(how="S")
    s = s.asfreq("MS")
    return s.astype(float)


# =========================================================
# MAIN RUN por codigo
# =========================================================
def run_for_codigo(codigo: str, s: pd.Series):
    s = s.dropna().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) < (max(LOOKBACK_NN, LAGS_TABULAR) + TEST_LEN + 1):
        print(f"[SKIP] {codigo}: muy corta (n={len(s)})")
        return None

    all_zero = is_all_zero_series(s)

    n = len(s)
    train_end_i = n - TEST_LEN
    train_end_date = s.index[train_end_i]

    y_train = s.iloc[:train_end_i]
    y_test  = s.iloc[train_end_i:]

    test_idx   = y_test.index
    future_idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(1), periods=H_FUTURE, freq="MS")

    preds_test: Dict[str, Optional[np.ndarray]] = {}
    preds_fut:  Dict[str, Optional[np.ndarray]] = {}
    scores: Dict[str, float] = {}

    if all_zero:
        for name in ["ETS","TCN","LSTM","DL_MultiTask","Linear","Ridge","MLP","SARIMAX","HGB"]:
            preds_test[name] = np.zeros(len(test_idx), float)
            preds_fut[name]  = np.zeros(len(future_idx), float)
            scores[name]     = 0.0
    else:
        if RUN_ETS:
            try:
                yhat_test, yhat_fut = fit_predict_ets(y_train, test_len=len(y_test), h_future=H_FUTURE)
                preds_test["ETS"] = yhat_test
                preds_fut["ETS"]  = yhat_fut
                scores["ETS"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} ETS falló: {e}")
                preds_test["ETS"] = None
                preds_fut["ETS"]  = None

        if RUN_SARIMAX:
            try:
                yhat_test, yhat_fut = fit_predict_sarimax(y_train, test_len=len(y_test), h_future=H_FUTURE)
                preds_test["SARIMAX"] = yhat_test
                preds_fut["SARIMAX"]  = yhat_fut
                scores["SARIMAX"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} SARIMAX falló: {e}")
                preds_test["SARIMAX"] = None
                preds_fut["SARIMAX"]  = None

        if RUN_LINEAR:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="Linear", use_log1p=False
                )
                preds_test["Linear"] = yhat_test
                preds_fut["Linear"]  = yhat_fut
                scores["Linear"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} Linear falló: {e}")
                preds_test["Linear"] = None
                preds_fut["Linear"]  = None

        if RUN_RIDGE:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="Ridge", use_log1p=False
                )
                preds_test["Ridge"] = yhat_test
                preds_fut["Ridge"]  = yhat_fut
                scores["Ridge"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} Ridge falló: {e}")
                preds_test["Ridge"] = None
                preds_fut["Ridge"]  = None

        if RUN_MLP:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="MLP", use_log1p=False
                )
                preds_test["MLP"] = yhat_test
                preds_fut["MLP"]  = yhat_fut
                scores["MLP"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} MLP falló: {e}")
                preds_test["MLP"] = None
                preds_fut["MLP"]  = None

        if RUN_HGB:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="HGB", use_log1p=True
                )
                preds_test["HGB"] = yhat_test
                preds_fut["HGB"]  = yhat_fut
                scores["HGB"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} HGB falló: {e}")
                preds_test["HGB"] = None
                preds_fut["HGB"]  = None

        sc = fit_asinh_scaler(y_train.values)
        y_scaled_all = transform_asinh(s.values, sc)

        X_all, Y_all = make_supervised(y_scaled_all, LOOKBACK_NN)
        target_idx = s.index[LOOKBACK_NN:]

        train_mask = target_idx < s.index[train_end_i]
        test_mask  = target_idx >= s.index[train_end_i]

        Xtr, Ytr = X_all[train_mask], Y_all[train_mask]
        Xte, Yte = X_all[test_mask],  Y_all[test_mask]
        Xte, Yte = Xte[:len(y_test)], Yte[:len(y_test)]

        ntr = len(Xtr)
        nval = max(10, int(0.1 * ntr))
        Xtrain, Ytrain = Xtr[:-nval], Ytr[:-nval]
        Xval,   Yval   = Xtr[-nval:], Ytr[-nval:]

        last_window = y_scaled_all[-LOOKBACK_NN:]

        if RUN_TCN:
            try:
                m_tcn = build_tcn(LOOKBACK_NN)
                train_keras_model(m_tcn, Xtrain, Ytrain, Xval, Yval, multitask=False)

                yhat_test_scaled = predict_keras_one_step(m_tcn, Xte)
                yhat_test = inverse_asinh(yhat_test_scaled, sc)

                yhat_fut_scaled = forecast_keras_autoregressive(m_tcn, last_window, H_FUTURE)
                yhat_fut = inverse_asinh(yhat_fut_scaled, sc)

                preds_test["TCN"] = yhat_test
                preds_fut["TCN"]  = yhat_fut
                scores["TCN"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} TCN falló: {e}")
                preds_test["TCN"] = None
                preds_fut["TCN"]  = None

        if RUN_LSTM:
            try:
                m_lstm = build_lstm(LOOKBACK_NN)
                train_keras_model(m_lstm, Xtrain, Ytrain, Xval, Yval, multitask=False)

                yhat_test_scaled = predict_keras_one_step(m_lstm, Xte)
                yhat_test = inverse_asinh(yhat_test_scaled, sc)

                yhat_fut_scaled = forecast_keras_autoregressive(m_lstm, last_window, H_FUTURE)
                yhat_fut = inverse_asinh(yhat_fut_scaled, sc)

                preds_test["LSTM"] = yhat_test
                preds_fut["LSTM"]  = yhat_fut
                scores["LSTM"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} LSTM falló: {e}")
                preds_test["LSTM"] = None
                preds_fut["LSTM"]  = None

        if RUN_MULTITASK_DL:
            try:
                m_mt = build_multitask_dl(LOOKBACK_NN)
                train_keras_model(m_mt, Xtrain, Ytrain, Xval, Yval, multitask=True)

                yhat_test_scaled = predict_keras_one_step(m_mt, Xte)
                yhat_test = inverse_asinh(yhat_test_scaled, sc)

                yhat_fut_scaled = forecast_keras_autoregressive(m_mt, last_window, H_FUTURE)
                yhat_fut = inverse_asinh(yhat_fut_scaled, sc)

                preds_test["DL_MultiTask"] = yhat_test
                preds_fut["DL_MultiTask"]  = yhat_fut
                scores["DL_MultiTask"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} MultiTask DL falló: {e}")
                preds_test["DL_MultiTask"] = None
                preds_fut["DL_MultiTask"]  = None

    # ===== imprimir ranking =====
    scored = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e99)
    print(f"\n=== {codigo} (n={len(s)}, test={len(y_test)}) ===")
    for name, r in scored:
        print(f"  {name:12s} RMSE_test = {r:,.4f}")

    # ===== outputs por-código =====
    out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
    ensure_dir(out_c_dir)

    out_png_all = os.path.join(out_c_dir, f"plot_{codigo}.png")
    out_png_fore = os.path.join(out_c_dir, f"plot_forecast_only_{codigo}.png")
    out_csv = os.path.join(out_c_dir, f"pred_{codigo}.csv")

    plot_all_models(
        codigo=codigo,
        y=s,
        train_end=train_end_date,
        test_idx=test_idx,
        future_idx=future_idx,
        preds_test=preds_test,
        preds_future=preds_fut,
        out_png=out_png_all
    )

    # NUEVA gráfica: solo forecast (sin test)
    plot_forecast_only(
        codigo=codigo,
        y=s,
        train_end=train_end_date,
        future_idx=future_idx,
        preds_future=preds_fut,
        out_png=out_png_fore,
        #tail_months=36
    )

    export_csv_codigo(
        out_csv=out_csv,
        y=s,
        test_idx=test_idx,
        future_idx=future_idx,
        preds_test=preds_test,
        preds_future=preds_fut
    )

    print(f"[OK] guardado: {out_png_all}")
    print(f"[OK] guardado: {out_png_fore}")
    print(f"[OK] guardado: {out_csv}")

    return {
        "codigo": codigo,
        "future_idx": future_idx,
        "preds_fut": preds_fut,
        "scores": scores,
    }


def main():
    ensure_dir(OUT_DIR)
    long = read_wide_monthly(CSV_PATH)

    codigos = sorted(long["codigo"].unique().tolist())
    if ONLY_CODIGO is not None:
        codigos = [str(ONLY_CODIGO)]

    print(f"[INFO] codigos a procesar: {len(codigos)}")
    print(f"[INFO] outputs: {OUT_DIR}")

    # ===== acumuladores para export ancho =====
    wide_by_model: Dict[str, Dict[str, Dict[str, float]]] = {}
    wide_best: Dict[str, Dict[str, float]] = {}
    rmse_by_model: Dict[str, List[float]] = {}
    global_future_idx: Optional[pd.DatetimeIndex] = None
    global_month_cols: Optional[List[str]] = None
    # ========================================

    for c in codigos:
        try:
            s = series_by_codigo(long, str(c))
            if s.dropna().empty:
                print(f"[SKIP] {c}: serie vacía")
                continue

            res = run_for_codigo(str(c), s)
            if res is None:
                continue

            future_idx = res["future_idx"]
            preds_fut = res["preds_fut"]
            scores = res.get("scores", {})
            codigo = res["codigo"]

            if global_future_idx is None:
                global_future_idx = future_idx
                global_month_cols = fmt_month_cols(global_future_idx)

            for mname, r in scores.items():
                if r is None:
                    continue
                if isinstance(r, float) and (not np.isfinite(r)):
                    continue
                rmse_by_model.setdefault(mname, []).append(float(r))

            for model_name, arr in preds_fut.items():
                if arr is None:
                    continue
                arr = np.asarray(arr, float).reshape(-1)
                m = min(len(arr), len(global_future_idx))

                wide_by_model.setdefault(model_name, {}).setdefault(codigo, {})
                for i in range(m):
                    wide_by_model[model_name][codigo][global_month_cols[i]] = float(arr[i])

            if len(scores) > 0:
                best_model = sorted(scores.items(), key=lambda kv: kv[1])[0][0]
                best_arr = preds_fut.get(best_model, None)
                if best_arr is not None:
                    best_arr = np.asarray(best_arr, float).reshape(-1)
                    m = min(len(best_arr), len(global_future_idx))
                    wide_best[codigo] = {}
                    for i in range(m):
                        wide_best[codigo][global_month_cols[i]] = float(best_arr[i])

        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            continue

    # ===== export final ancho =====
    if EXPORT_WIDE_FUTURE and global_future_idx is not None and global_month_cols is not None:
        ensure_dir(OUT_DIR)

        for model_name, rows in wide_by_model.items():
            out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_{model_name}.csv")
            export_wide_future_csv(out_path, rows, global_month_cols)
            print(f"[OK] guardado ancho: {out_path}")

        if len(wide_best) > 0:
            out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_BEST.csv")
            export_wide_future_csv(out_path, wide_best, global_month_cols)
            print(f"[OK] guardado ancho: {out_path}")

        def pick_best_global(stat: str) -> Tuple[Optional[str], float]:
            best_model = None
            best_score = float("inf")
            for mname, vals in rmse_by_model.items():
                if len(vals) == 0:
                    continue
                score = float(np.mean(vals)) if stat == "mean" else float(np.median(vals))
                if score < best_score:
                    best_score = score
                    best_model = mname
            return best_model, best_score

        best_mean_model, best_mean_score = pick_best_global("mean")
        if best_mean_model is not None and best_mean_model in wide_by_model:
            out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_BEST_GLOBAL_MEAN.csv")
            export_wide_future_csv(out_path, wide_by_model[best_mean_model], global_month_cols)
            print(f"[OK] guardado ancho: {out_path}  (BEST_GLOBAL_MEAN={best_mean_model}, RMSE_mean={best_mean_score:,.4f})")

        best_med_model, best_med_score = pick_best_global("median")
        if best_med_model is not None and best_med_model in wide_by_model:
            out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_BEST_GLOBAL_MEDIAN.csv")
            export_wide_future_csv(out_path, wide_by_model[best_med_model], global_month_cols)
            print(f"[OK] guardado ancho: {out_path}  (BEST_GLOBAL_MEDIAN={best_med_model}, RMSE_median={best_med_score:,.4f})")


if __name__ == "__main__":
    main()
