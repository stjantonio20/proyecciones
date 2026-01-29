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
import matplotlib
matplotlib.use("Agg")  # backend sin GUI (evita Tkinter)
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
WIDE_PREFIX = "proyeccion"  # nombre base de archivos
# ====================================

CSV_PATH = "./dataset/Crediguate_actualizado_mensual.csv"
OUT_DIR  = "outputs_modelos_37meses"

H_FUTURE = 37           # meses a pronosticar
TEST_LEN = 6           # meses test fijo

#ONLY_CODIGO = "101101" # o None para todos
ONLY_CODIGO = None

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

RUN_MLP_MIMO_LOG  = True   # MIMO MLP en log1p + calendario (multi-step directo)
RUN_FOURIER_RIDGE = True   # Ridge Fourier (log1p) + tendencia (robusto)

# --- heurísticas para series con muchos ceros / outliers ---
ZERO_HEAVY_RATIO = 0.70     # si >=70% de puntos son 0 => recortar train (no-NN) desde año del 1er no-cero
EXAG_FACTOR      = 6.0      # si forecast supera (factor * escala reciente) se oculta en la gráfica
EXAG_TAIL_MONTHS = 24       # ventana reciente para medir escala
LOOKBACK_MIMO    = 24       # lookback para MIMO MLP (en meses)

# NUEVOS: multistep + level+growth (log1p)
RUN_TCN_MS_LG    = True
RUN_LSTM_MS_LG   = True

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
# Scaling robusto (asinh + Standard) para NNs (legacy)
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
# Windows supervisadas (NNs) 1-step (legacy)
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
# Heurísticas: series con muchos 0s + filtro de forecasts exagerados
# =========================================================
def is_zero_heavy(s: pd.Series, ratio: float = ZERO_HEAVY_RATIO) -> bool:
    v = pd.to_numeric(s.values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return True
    return float(np.mean(v == 0.0)) >= float(ratio)

def trim_series_from_first_nonzero_year(s: pd.Series) -> pd.Series:
    """
    Si la serie tiene MUCHOS ceros, recorta desde Enero del año
    donde aparece el primer valor != 0 (manteniendo meses previos del mismo año).
    """
    s = s.copy().astype(float)
    nz = s[s != 0.0]
    if nz.empty:
        return s
    first_dt = nz.index[0]
    start_dt = pd.Timestamp(year=first_dt.year, month=1, day=1)
    if start_dt < s.index[0]:
        start_dt = s.index[0]
    return s.loc[start_dt:]

def forecast_is_exaggerated(pred: np.ndarray, s_hist: pd.Series) -> bool:
    """
    Heurística robusta para ocultar modelos que explotan (ej. TCN con picos enormes).
    No elimina el modelo del CSV; solo evita que arruine la gráfica.
    """
    if pred is None:
        return False
    p = np.asarray(pred, float).reshape(-1)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return False

    tail = s_hist.dropna().astype(float)
    if tail.empty:
        return False
    tail = tail.iloc[-min(len(tail), EXAG_TAIL_MONTHS):]

    # escala robusta reciente
    ref = float(np.nanmedian(np.abs(tail.values)))
    if (not np.isfinite(ref)) or ref <= 0:
        ref = float(np.nanmax(np.abs(tail.values))) if np.isfinite(np.nanmax(np.abs(tail.values))) else 1.0
        if ref <= 0:
            ref = 1.0

    # condición 1: amplitud absoluta demasiado grande
    if float(np.nanmax(np.abs(p))) > EXAG_FACTOR * ref:
        return True

    # condición 2: saltos mes-a-mes demasiado agresivos vs. historia reciente
    dif_tail = np.diff(tail.values)
    mad = float(np.nanmedian(np.abs(dif_tail - np.nanmedian(dif_tail))))
    if (not np.isfinite(mad)) or mad <= 0:
        mad = float(np.nanmedian(np.abs(dif_tail))) if dif_tail.size else 0.0

    if mad > 0:
        dif_p = np.diff(p)
        if dif_p.size and float(np.nanmax(np.abs(dif_p))) > (EXAG_FACTOR * 3.0) * mad:
            return True

    return False

# =========================================================
# NUEVO: features level+growth (log1p) + calendario
# =========================================================
def build_level_growth_features(y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    level: z_t = log1p(max(y_t,0))
    growth: g_t = z_t - z_{t-1}  (0 para t=0)
    return z (n,), g (n,)
    """
    yv = np.asarray(y.values, float)
    yv = np.maximum(yv, 0.0)
    z = np.log1p(yv)
    g = np.zeros_like(z)
    if len(z) >= 2:
        g[1:] = z[1:] - z[:-1]
    return z, g

def fit_standard_2d(z: np.ndarray, g: np.ndarray, train_n: int) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    z_tr = z[:train_n]
    g_tr = g[:train_n]
    z_mu, z_sd = float(np.nanmean(z_tr)), float(np.nanstd(z_tr))
    g_mu, g_sd = float(np.nanmean(g_tr)), float(np.nanstd(g_tr))
    if (not np.isfinite(z_sd)) or z_sd == 0: z_sd = 1.0
    if (not np.isfinite(g_sd)) or g_sd == 0: g_sd = 1.0
    return (z_mu, z_sd), (g_mu, g_sd)

def transform_standard(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / sd

def inverse_standard(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return x * sd + mu

def make_supervised_multistep_features(
    feat_mat: np.ndarray,   # (n, d)
    target_z: np.ndarray,   # (n,)
    lookback: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye dataset multistep:
      X[i] = feat[t-lookback:t, :]
      Y[i] = target_z[t:t+horizon]
    donde t recorre desde lookback hasta n-horizon
    retorna:
      X: (m, lookback, d)
      Y: (m, horizon)
      t_idx: (m,) índices t (inicio de Y)
    """
    n = len(target_z)
    X, Y, T = [], [], []
    for t in range(lookback, n - horizon + 1):
        X.append(feat_mat[t - lookback:t, :])
        Y.append(target_z[t:t + horizon])
        T.append(t)
    return np.asarray(X, float), np.asarray(Y, float), np.asarray(T, int)


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
# MIMO MLP (log1p + features calendario) y Fourier-Ridge (robusto)
# =========================================================
def calendar_features_abs(dates: pd.DatetimeIndex, t_abs: np.ndarray) -> np.ndarray:
    """sin/cos mes + sin/cos anual usando 't_abs' (meses desde inicio)."""
    month = dates.month.values.astype(float)
    sin_m = np.sin(2 * np.pi * month / 12.0)
    cos_m = np.cos(2 * np.pi * month / 12.0)

    t_abs = np.asarray(t_abs, float).reshape(-1)
    sin_t = np.sin(2 * np.pi * t_abs / 12.0)
    cos_t = np.cos(2 * np.pi * t_abs / 12.0)
    return np.column_stack([sin_m, cos_m, sin_t, cos_t])

def make_supervised_mimo_log(y: pd.Series, lookback: int, horizon: int):
    """
    X = [log1p(y_{t-lookback:t-1}), calendar_features(t)]
    Y = [log1p(y_{t:t+horizon-1})]
    """
    y = y.copy().astype(float)
    y_log = np.log1p(np.maximum(y.values, 0.0))  # si hay negativos, se truncan a 0 para esta ruta
    idx = y.index
    t_abs_all = np.arange(len(idx), dtype=float)
    cal = calendar_features_abs(idx, t_abs_all)

    X_list, Y_list, t_index = [], [], []
    for t in range(lookback, len(y) - horizon + 1):
        x_lags = y_log[t - lookback:t]
        x_cal = cal[t]
        X = np.concatenate([x_lags, x_cal], axis=0)
        Y = y_log[t:t + horizon]
        X_list.append(X)
        Y_list.append(Y)
        t_index.append(idx[t])

    return np.asarray(X_list, float), np.asarray(Y_list, float), pd.DatetimeIndex(t_index)

def build_mimo_mlp_model(horizon: int, random_state: int = 24):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    # Menos agresivo: más regularización + LR más bajo + early stopping
    base = MLPRegressor(
        hidden_layer_sizes=(96, 48),
        activation="relu",
        solver="adam",
        alpha=5e-3,
        learning_rate_init=5e-4,
        max_iter=1500,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=40,
        validation_fraction=0.2,
    )
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mimo", MultiOutputRegressor(base))
    ])
    return model

def fit_predict_mimo_mlp_log(
    s_full: pd.Series,
    s_train: pd.Series,
    test_len: int,
    h_future: int,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    - Eval: entrena en s_train y predice un bloque de H=max(test_len, h_future) desde train_end
    - Forecast: re-entrena con s_full y predice h_future desde el final
    """
    H = int(max(test_len, h_future))

    # ---------- evaluación (test) ----------
    Xtr, Ytr, _ = make_supervised_mimo_log(s_train, lookback=lookback, horizon=H)
    if len(Xtr) < 20:
        raise ValueError("MIMO_MLP_LOG: muy pocos ejemplos supervisados (<20).")
    model = build_mimo_mlp_model(horizon=H, random_state=24)
    model.fit(Xtr, Ytr)

    # construir x en train_end (inicio de test)
    # usamos log1p(max(.,0)) por esta ruta; si hay negativos, este modelo NO aplica bien.
    ylog = np.log1p(np.maximum(s_train.values.astype(float), 0.0))
    L = lookback
    x_lags = ylog[-L:]
    # t_abs del punto train_end en s_train es len(s_train)-1, pero features usan t del inicio del forecast:
    start_date = s_train.index[-1] + pd.offsets.MonthBegin(1)
    # t_abs para start_date = len(s_train)
    x_cal = calendar_features_abs(pd.DatetimeIndex([start_date]), np.array([len(s_train)], float)).ravel()
    x = np.concatenate([x_lags, x_cal], axis=0).reshape(1, -1)

    yhat_log_block = model.predict(x)[0]
    yhat_block = np.expm1(yhat_log_block)

    yhat_test = yhat_block[:test_len]

    # ---------- forecast final ----------
    Xall, Yall, _ = make_supervised_mimo_log(s_full, lookback=lookback, horizon=H)
    if len(Xall) < 20:
        raise ValueError("MIMO_MLP_LOG: muy pocos ejemplos (full) (<20).")
    model2 = build_mimo_mlp_model(horizon=H, random_state=24)
    model2.fit(Xall, Yall)

    ylog2 = np.log1p(np.maximum(s_full.values.astype(float), 0.0))
    x_lags2 = ylog2[-L:]
    start_date2 = s_full.index[-1] + pd.offsets.MonthBegin(1)
    x_cal2 = calendar_features_abs(pd.DatetimeIndex([start_date2]), np.array([len(s_full)], float)).ravel()
    x2 = np.concatenate([x_lags2, x_cal2], axis=0).reshape(1, -1)

    yhat_log_fut = model2.predict(x2)[0]
    yhat_fut = np.expm1(yhat_log_fut)[:h_future]

    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)

def fourier_design_abs(n_hist: int, future_h: int, K: int):
    """Intercepto + tendencia + K armónicos anuales, con t absoluto."""
    t = np.arange(n_hist, dtype=float)
    X = [np.ones(n_hist), t]
    for k in range(1, K + 1):
        X.append(np.sin(2 * np.pi * k * t / 12.0))
        X.append(np.cos(2 * np.pi * k * t / 12.0))
    X = np.column_stack(X)

    tf = np.arange(n_hist, n_hist + future_h, dtype=float)
    Xf = [np.ones(future_h), tf]
    for k in range(1, K + 1):
        Xf.append(np.sin(2 * np.pi * k * tf / 12.0))
        Xf.append(np.cos(2 * np.pi * k * tf / 12.0))
    Xf = np.column_stack(Xf)
    return X, Xf

def fit_predict_fourier_ridge_log(y_train: pd.Series, test_len: int, h_future: int, K: int = 3, alpha: float = 5.0):
    """Ridge sobre Fourier+tendencia en log1p; robusto y no suele explotar."""
    from sklearn.linear_model import Ridge

    ylog = np.log1p(np.maximum(y_train.values.astype(float), 0.0))
    X, Xf = fourier_design_abs(len(y_train), test_len + h_future, K=K)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, ylog)

    yhat_log = reg.predict(Xf)
    yhat = np.expm1(yhat_log)
    yhat_test = yhat[:test_len]
    yhat_fut = yhat[-h_future:]
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
# Keras builders: TCN / LSTM / MultiTask DL (legacy 1-step)
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

# =========================================================
# NUEVO: Keras builders multistep con features (level+growth+cal)
# =========================================================
def build_lstm_multistep(input_len: int, n_feat: int, horizon: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_len, n_feat))
    x = layers.LSTM(96, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(48)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(horizon)(x)  # Dense(H) -> multistep directo
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
    return m

def build_tcn_multistep(input_len: int, n_feat: int, horizon: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    try:
        from tcn import TCN
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
        from tcn import TCN

    inp = layers.Input(shape=(input_len, n_feat))
    x = TCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
        padding="causal",
        dropout_rate=0.2,
        return_sequences=False,
        use_skip_connections=True,
    )(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(horizon)(x)  # Dense(H)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
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

def save_future_preds_wide_excel(
    out_xlsx: str,
    results: List[dict],
    month_cols: Optional[List[str]] = None,
    sheet_name: str = "wide_future",
):
    if not results:
        raise ValueError("results está vacío. No hay nada que exportar.")

    if month_cols is None:
        fut0 = results[0].get("future_idx", None)
        if fut0 is None:
            raise ValueError("El primer elemento de results no tiene 'future_idx'.")
        fut0 = pd.DatetimeIndex(fut0)
        month_cols = [d.strftime("%b-%y") for d in fut0]

    rows = []
    for res in results:
        codigo = str(res.get("codigo", ""))
        fut_idx = res.get("future_idx", None)
        preds_fut = res.get("preds_fut", {}) or {}

        if fut_idx is None:
            continue
        fut_idx = pd.DatetimeIndex(fut_idx)
        local_month_cols = [d.strftime("%b-%y") for d in fut_idx]

        for model_name, arr in preds_fut.items():
            if arr is None:
                continue
            arr = np.asarray(arr, float).reshape(-1)
            m = min(len(arr), len(local_month_cols))

            row = {"codigo": codigo, "modelo": str(model_name)}
            for i in range(m):
                row[local_month_cols[i]] = float(arr[i]) if np.isfinite(arr[i]) else np.nan

            rows.append(row)

    if not rows:
        raise ValueError("No se generaron filas (quizá preds_fut está vacío o todo es None).")

    df = pd.DataFrame(rows)

    fixed = ["codigo", "modelo"]
    for c in month_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[fixed + month_cols]
    df["codigo_ord"] = pd.to_numeric(df["codigo"], errors="coerce")
    df = df.sort_values(["codigo_ord", "codigo", "modelo"]).drop(columns=["codigo_ord"])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    return df

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
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Real", linewidth=2)

    if train_end is not None:
        plt.axvline(train_end, linestyle="--", linewidth=1)
        y_min = np.nanmin(y.values) if np.isfinite(np.nanmin(y.values)) else 0.0
        plt.text(train_end, y_min, "  train_end", rotation=90, va="bottom")

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
# =========================================================
# MAIN RUN por codigo
# =========================================================
def run_for_codigo(codigo: str, s: pd.Series):

    # ---------- limpieza base ----------
    s = s.dropna().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    s_full = s.copy()  # serie completa para plots

    # ---------- casos especiales ----------
    all_zero = is_all_zero_series(s_full)
    zero_heavy = is_zero_heavy(s_full) and (not all_zero)

    # ---- caso 1: TODO CEROS ----
    if all_zero:
        print(f"[INFO] {codigo}: serie completamente cero")

        future_idx = pd.date_range(
            start=s_full.index[-1] + pd.offsets.MonthBegin(1),
            periods=H_FUTURE,
            freq="MS"
        )

        preds_test = {}
        preds_fut = {}
        scores = {}

        for name in [
            "ETS","SARIMAX","Linear","Ridge","MLP","HGB",
            "TCN","LSTM","DL_MultiTask",
            "TCN_MS_LG","LSTM_MS_LG","FourierRidge","MLP_MIMO_LOG"
        ]:
            preds_test[name] = np.zeros(TEST_LEN)
            preds_fut[name]  = np.zeros(H_FUTURE)
            scores[name] = 0.0

        # outputs mínimos
        out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
        ensure_dir(out_c_dir)

        plot_forecast_only(
            codigo=codigo,
            y=s_full,
            train_end=None,
            future_idx=future_idx,
            preds_future=preds_fut,
            out_png=os.path.join(out_c_dir, f"plot_forecast_only_{codigo}.png")
        )

        return {
            "codigo": codigo,
            "future_idx": future_idx,
            "preds_fut": preds_fut,
            "scores": scores,
        }

    # ---- caso 2: MUCHOS CEROS AL INICIO ----
    if zero_heavy:
        print(f"[INFO] {codigo}: zero-heavy, recortando serie")
        s = trim_series_from_first_nonzero_year(s_full)

    # ---------- validación de longitud ----------
    # Regla normal (para tabulares con muchos lags y NNs)
    min_len_normal = max(LOOKBACK_NN, LAGS_TABULAR) + TEST_LEN + 5

    # Regla especial para zero-heavy: permitimos series cortas, pero al menos 12m + test
    min_len_zeroheavy = 12 + TEST_LEN + 1

    min_len = min_len_zeroheavy if zero_heavy else min_len_normal

    if len(s) < min_len:
        print(f"[SKIP] {codigo}: muy corta (n={len(s)}) (min={min_len}, zero_heavy={zero_heavy})")
        return None


    # ---------- split ----------
    n = len(s)
    train_end_i = n - TEST_LEN
    train_end_date = s.index[train_end_i]

    y_train = s.iloc[:train_end_i]
    y_test  = s.iloc[train_end_i:]

    test_idx = y_test.index
    future_idx = pd.date_range(
        start=s.index[-1] + pd.offsets.MonthBegin(1),
        periods=H_FUTURE,
        freq="MS"
    )
    # ---------- lags locales (para no matar series cortas) ----------
    if zero_heavy:
        # con 23 meses, LAGS_TABULAR=28 es imposible, bajamos a algo razonable
        local_lags_tab = min(LAGS_TABULAR, max(6, len(y_train) - 1))
    else:
        local_lags_tab = LAGS_TABULAR

    # ---------- switches locales ----------
    if zero_heavy:
        local_run_tcn = False
        local_run_lstm = False
        local_run_multitask = False
        local_run_tcn_ms_lg = False
        local_run_lstm_ms_lg = False
    else:
        local_run_tcn = RUN_TCN
        local_run_lstm = RUN_LSTM
        local_run_multitask = RUN_MULTITASK_DL
        local_run_tcn_ms_lg = RUN_TCN_MS_LG
        local_run_lstm_ms_lg = RUN_LSTM_MS_LG

    # ---------- contenedores ----------
    preds_test: Dict[str, Optional[np.ndarray]] = {}
    preds_fut:  Dict[str, Optional[np.ndarray]] = {}
    scores: Dict[str, float] = {}


    if all_zero:
        for name in ["ETS","TCN","LSTM","DL_MultiTask","Linear","Ridge","MLP","SARIMAX","HGB","TCN_MS_LG","LSTM_MS_LG"]:
            preds_test[name] = np.zeros(len(test_idx), float)
            preds_fut[name]  = np.zeros(len(future_idx), float)
            scores[name]     = 0.0
    else:
        # ============ ETS ============
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

        # ============ SARIMAX ============
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

        # ============ LINEAR ============
        if RUN_LINEAR:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=local_lags_tab, model_kind="Linear", use_log1p=False
                )

                preds_test["Linear"] = yhat_test
                preds_fut["Linear"]  = yhat_fut
                scores["Linear"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} Linear falló: {e}")
                preds_test["Linear"] = None
                preds_fut["Linear"]  = None

        # ============ RIDGE ============
        if RUN_RIDGE:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                lags=local_lags_tab, model_kind="Linear", use_log1p=False
                )
                preds_test["Ridge"] = yhat_test
                preds_fut["Ridge"]  = yhat_fut
                scores["Ridge"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} Ridge falló: {e}")
                preds_test["Ridge"] = None
                preds_fut["Ridge"]  = None

        # ============ MLP ============
        if RUN_MLP:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=local_lags_tab, model_kind="Linear", use_log1p=False
                )
                preds_test["MLP"] = yhat_test
                preds_fut["MLP"]  = yhat_fut
                scores["MLP"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} MLP falló: {e}")
                preds_test["MLP"] = None
                preds_fut["MLP"]  = None

        # ============ HGB ============
        if RUN_HGB:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    s, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=local_lags_tab, model_kind="Linear", use_log1p=False
                )
                preds_test["HGB"] = yhat_test
                preds_fut["HGB"]  = yhat_fut
                scores["HGB"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} HGB falló: {e}")
                preds_test["HGB"] = None
                preds_fut["HGB"]  = None

        # =====================================================
        # NUEVO: TCN/LSTM multistep con (level+growth+cal)
        #   - predice en espacio z=log1p(y) -> luego invierte expm1
        #   - salida Dense(H_OUT) (multi-step directo)
        # =====================================================
        try:
            z, g = build_level_growth_features(s)
            H_OUT = int(max(TEST_LEN, H_FUTURE))  # salida multistep suficientemente larga
            (z_mu, z_sd), (g_mu, g_sd) = fit_standard_2d(z, g, train_n=train_end_i)

            z_s = transform_standard(z, z_mu, z_sd)
            g_s = transform_standard(g, g_mu, g_sd)

            cal = month_features(s.index)  # (n,2) sin/cos
            feat = np.c_[z_s, g_s, cal]    # (n,4)

            X_ms, Y_ms, T_ms = make_supervised_multistep_features(
                feat_mat=feat,
                target_z=z_s,      # target en z escalado
                lookback=LOOKBACK_NN,
                horizon=H_OUT
            )

            # máscara de train: el primer paso del target (t) debe caer antes de train_end_i
            train_mask = (T_ms < train_end_i)
            Xtr_ms, Ytr_ms = X_ms[train_mask], Y_ms[train_mask]

            if len(Xtr_ms) >= 30:
                ntr = len(Xtr_ms)
                nval = max(10, int(0.1 * ntr))
                Xtrain_ms, Ytrain_ms = Xtr_ms[:-nval], Ytr_ms[:-nval]
                Xval_ms,   Yval_ms   = Xtr_ms[-nval:], Ytr_ms[-nval:]

                # ventana para TEST: termina justo antes del primer test (t=train_end_i)
                x_test_win = feat[train_end_i - LOOKBACK_NN:train_end_i, :].reshape(1, LOOKBACK_NN, feat.shape[1])

                # ventana para FUTURE: termina al final de la serie
                x_future_win = feat[-LOOKBACK_NN:, :].reshape(1, LOOKBACK_NN, feat.shape[1])

                if local_run_tcn_ms_lg:
                    try:
                        m = build_tcn_multistep(LOOKBACK_NN, feat.shape[1], H_OUT)
                        train_keras_model(m, Xtrain_ms, Ytrain_ms, Xval_ms, Yval_ms, multitask=False)

                        # TEST: predice próximos H_OUT z
                        zhat_test_s = np.asarray(m.predict(x_test_win, verbose=0)).reshape(-1)[:TEST_LEN]
                        zhat_test = inverse_standard(zhat_test_s, z_mu, z_sd)
                        yhat_test = np.expm1(zhat_test)

                        # FUTURE: próximos H_OUT desde último punto (tomamos H_FUTURE)
                        zhat_fut_s = np.asarray(m.predict(x_future_win, verbose=0)).reshape(-1)[:H_FUTURE]
                        zhat_fut = inverse_standard(zhat_fut_s, z_mu, z_sd)
                        yhat_fut = np.expm1(zhat_fut)

                        preds_test["TCN_MS_LG"] = yhat_test
                        preds_fut["TCN_MS_LG"]  = yhat_fut
                        scores["TCN_MS_LG"] = rmse(y_test.values, yhat_test)
                    except Exception as e:
                        print(f"[WARN] {codigo} TCN_MS_LG falló: {e}")
                        preds_test["TCN_MS_LG"] = None
                        preds_fut["TCN_MS_LG"]  = None

                if local_run_lstm_ms_lg:
                    try:
                        m = build_lstm_multistep(LOOKBACK_NN, feat.shape[1], H_OUT)
                        train_keras_model(m, Xtrain_ms, Ytrain_ms, Xval_ms, Yval_ms, multitask=False)

                        zhat_test_s = np.asarray(m.predict(x_test_win, verbose=0)).reshape(-1)[:TEST_LEN]
                        zhat_test = inverse_standard(zhat_test_s, z_mu, z_sd)
                        yhat_test = np.expm1(zhat_test)

                        zhat_fut_s = np.asarray(m.predict(x_future_win, verbose=0)).reshape(-1)[:H_FUTURE]
                        zhat_fut = inverse_standard(zhat_fut_s, z_mu, z_sd)
                        yhat_fut = np.expm1(zhat_fut)

                        preds_test["LSTM_MS_LG"] = yhat_test
                        preds_fut["LSTM_MS_LG"]  = yhat_fut
                        scores["LSTM_MS_LG"] = rmse(y_test.values, yhat_test)
                    except Exception as e:
                        print(f"[WARN] {codigo} LSTM_MS_LG falló: {e}")
                        preds_test["LSTM_MS_LG"] = None
                        preds_fut["LSTM_MS_LG"]  = None
            else:
                preds_test["TCN_MS_LG"] = None
                preds_fut["TCN_MS_LG"]  = None
                preds_test["LSTM_MS_LG"] = None
                preds_fut["LSTM_MS_LG"]  = None
        except Exception as e:
            print(f"[WARN] {codigo} fallo preparando multistep level+growth: {e}")
            preds_test["TCN_MS_LG"] = None
            preds_fut["TCN_MS_LG"]  = None
            preds_test["LSTM_MS_LG"] = None
            preds_fut["LSTM_MS_LG"]  = None

        # =====================================================
        # Legacy NNs 1-step (asinh) - se mantienen
        # =====================================================

        # ============ FourierRidge (log1p) ============
        if RUN_FOURIER_RIDGE:
            try:
                yhat_test, yhat_fut = fit_predict_fourier_ridge_log(y_train, test_len=len(y_test), h_future=H_FUTURE, K=3, alpha=8.0)
                preds_test["FourierRidge"] = yhat_test
                preds_fut["FourierRidge"]  = yhat_fut
                scores["FourierRidge"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} FourierRidge falló: {e}")
                preds_test["FourierRidge"] = None
                preds_fut["FourierRidge"]  = None

        # ============ MLP_MIMO_LOG (multi-step directo) ============
        # Nota: este modelo trabaja en log1p(max(y,0)). Si la serie es mayormente negativa,
        # probablemente no aplique bien (por eso lo dejamos opcional).
        if RUN_MLP_MIMO_LOG:
            try:
                yhat_test, yhat_fut = fit_predict_mimo_mlp_log(
                    s_full=s,          # ya recortada si zero_heavy
                    s_train=y_train,
                    test_len=len(y_test),
                    h_future=H_FUTURE,
                    lookback=min(LOOKBACK_MIMO, max(8, len(y_train) - 5))
                )
                preds_test["MLP_MIMO_LOG"] = yhat_test
                preds_fut["MLP_MIMO_LOG"]  = yhat_fut
                scores["MLP_MIMO_LOG"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} MLP_MIMO_LOG falló: {e}")
                preds_test["MLP_MIMO_LOG"] = None
                preds_fut["MLP_MIMO_LOG"]  = None
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

        # ============ TCN ============
        if local_run_tcn:
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

        # ============ LSTM ============
        if local_run_lstm:
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

        if local_run_multitask:
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


    # ---- Filtrar forecasts exagerados SOLO para graficar (no afecta CSV) ----
    preds_test_plot = dict(preds_test)

    preds_fut_plot = {}
    for name, arr in preds_fut.items():
        if arr is None:
            continue
        if forecast_is_exaggerated(arr, s_full):
            print(f"[INFO] {codigo} ocultando en gráfica (exagerado): {name}")
            continue
        preds_fut_plot[name] = arr

    # ====== Graficar (UNA sola vez) ======
    plot_all_models(
        codigo=codigo,
        y=s_full,
        train_end=train_end_date,
        test_idx=test_idx,
        future_idx=future_idx,
        preds_test=preds_test_plot,
        preds_future=preds_fut_plot,
        out_png=out_png_all
    )

    plot_forecast_only(
        codigo=codigo,
        y=s_full,
        train_end=train_end_date,
        future_idx=future_idx,
        preds_future=preds_fut_plot,
        out_png=out_png_fore,
    )

    # ====== Export CSV (con TODOS los modelos, incluso los exagerados) ======
    export_csv_codigo(
        out_csv=out_csv,
        y=s_full,
        test_idx=test_idx,
        future_idx=future_idx,
        preds_test=preds_test_plot,
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

    wide_by_model: Dict[str, Dict[str, Dict[str, float]]] = {}
    wide_best: Dict[str, Dict[str, float]] = {}
    rmse_by_model: Dict[str, List[float]] = {}
    global_future_idx: Optional[pd.DatetimeIndex] = None
    global_month_cols: Optional[List[str]] = None

    all_results = []
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

            all_results.append({
                "codigo": res["codigo"],
                "future_idx": res["future_idx"],
                "preds_fut": res["preds_fut"],
            })
        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            continue

    out_xlsx = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_ALL_MODELS.xlsx")
    if not all_results:
        print("[WARN] No hubo resultados (todos los códigos fueron SKIP). No se exporta Excel/CSVs wide.")
        return
    save_future_preds_wide_excel(out_xlsx, all_results)
    print(f"[OK] guardado Excel wide: {out_xlsx}")

    if EXPORT_WIDE_FUTURE and global_future_idx is not None and global_month_cols is not None:
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

    plt.close("all")

if __name__ == "__main__":
    main()
