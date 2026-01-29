#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediccion_mensual_independiente_robusto_walkforward.py

VERSIÓN "walk-forward por mes" (mensual-independiente) manteniendo la MISMA estructura
de outputs del script robusto que sí te funcionaba.

Qué cambia vs tu script robusto anterior:
- En vez de llamar predict_one_month_target() para CADA fecha (y repetir trabajo),
  aquí trabajamos "por mes":
    * Extraemos la subserie del mes (eje = años) UNA vez.
    * Hacemos backtest walk-forward (solo para los meses del test).
    * Hacemos forecast futuro por mes, generando los años necesarios.
- Mantiene TODOS los modelos (ETS, SARIMAX, Linear, Ridge, MLP, HGB, TCN, LSTM, DL_MultiTask).
- Mantiene la mezcla nivel + crecimiento (YoY) como antes.
- Mantiene los mismos outputs:
    - por código: plot_all, plot_forecast_only, csv por-código
    - OUT_DIR/_solo_real_forecast/<codigo>.png
    - export wide por modelo + BEST + BEST_GLOBAL_MEAN/MEDIAN + Excel ALL_MODELS

IMPORTANTE sobre "mismos resultados":
- Para ETS/SARIMAX y tabulares (Linear/Ridge/MLP/HGB), el forecast FUTURO usa la MISMA
  base de entrenamiento (años reales) para cada target_year, igual que el script anterior,
  así que te debe dar resultados equivalentes (a redondeo numérico).
- Para NN (LSTM/TCN/DL_MultiTask), por velocidad entrenamos 1 sola vez por (código, mes, modelo)
  con TODO el historial real del mes y luego hacemos predicción recursiva de años futuros.
  Esto puede diferir ligeramente del script anterior (que re-entrenaba por cada target_year),
  pero mantiene el comportamiento "aprende la forma del mes" y acelera muchísimo.

Requisitos:
  pip install pandas numpy openpyxl scikit-learn statsmodels matplotlib
  (opcional NN) pip install tensorflow keras-tcn

Dataset esperado:
  CSV ancho con columnas: codigo, Mar-15, Apr-15, ..., Nov-25 (o similar, igual que tu primer script)
"""

import os
import re
import math
import warnings
import shutil
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
# CONFIG (ajusta rutas)
# =========================================================
EXPORT_WIDE_FUTURE = True
WIDE_PREFIX = "proyeccion"

CSV_PATH = "./dataset/Crediguate_actualizado_mensual.csv"
OUT_DIR  = "proyeccion_13meses_mensual_independiente_robusto_walkforward"
OUT_DIR_ONLY_FORECAST = os.path.join(OUT_DIR, "_solo_real_forecast")

H_FUTURE = 37          # si tu último dato es Nov-25 => futuro: Dec-25..Dec-28 (37 meses)
TEST_LEN = 6           # backtest: últimos 6 meses

ONLY_CODIGO = None     # "601101" o None

# crecimiento (YoY) lags cortos
GROWTH_LAGS = 2

# Pesos por recencia (como tu último script)
RECENT_YEARS = 10
HALF_LIFE_YEARS = 4.0

# switches
RUN_ETS          = True
RUN_TCN          = True
RUN_LSTM         = True
RUN_MULTITASK_DL = True

RUN_LINEAR       = True
RUN_RIDGE        = True
RUN_MLP          = True
RUN_SARIMAX      = True
RUN_HGB          = True

# NN params
NN_EPOCHS   = 180   # más bajo para velocidad (ajusta si quieres)
NN_BATCH    = 16
NN_PATIENCE = 10

# mezcla nivel + crecimiento (YoY mismo mes)
GROWTH_BLEND_W = 0.35
EPS_DEN = 1e-9

# =========================================================
# Utils: parse columnas tipo "Mar-15"
# =========================================================
MONTH_MAP = {
    # English
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    # Spanish (common LATAM)
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Sep": 9, "Set": 9, "Oct": 10, "Nov": 11, "Dic": 12
}

def parse_month_col(col: str) -> pd.Timestamp:
    col = str(col).strip()
    parts = col.split("-")
    if len(parts) != 2:
        raise ValueError(f"Columna de mes inválida: {col!r}")
    mon, yy = parts[0].strip(), parts[1].strip()
    mon = mon[:3].title()
    if len(yy) == 2:
        year = 2000 + int(yy)
    else:
        year = int(yy)
    if mon not in MONTH_MAP:
        raise ValueError(f"Mes inválido en columna {col!r} (mon={mon!r})")
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
# Export wide Excel: codigo|modelo|meses...
# =========================================================
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

# =========================================================
# Growth + blend
# =========================================================
def growth_yoy(y_year: pd.Series) -> pd.Series:
    y = y_year.copy().astype(float)
    yrs = y.index.values
    g = pd.Series(index=y.index, dtype=float)
    for yr in yrs:
        prev = int(yr) - 1
        if prev in y.index and np.isfinite(y.loc[prev]) and np.isfinite(y.loc[yr]):
            den = max(abs(float(y.loc[prev])), EPS_DEN)
            g.loc[yr] = (float(y.loc[yr]) - float(y.loc[prev])) / den
        else:
            g.loc[yr] = np.nan
    return g

def blend_level_and_growth(y_level_pred: float, g_pred: float, y_last: float) -> float:
    if not np.isfinite(y_level_pred):
        y_level_pred = np.nan
    if not np.isfinite(g_pred):
        g_pred = np.nan

    y_growth_pred = np.nan
    if np.isfinite(y_last) and np.isfinite(g_pred):
        y_growth_pred = float(y_last) * (1.0 + float(g_pred))

    w = float(GROWTH_BLEND_W)
    if (not np.isfinite(y_last)) or abs(float(y_last)) < 1.0:
        w = min(w, 0.15)

    if np.isfinite(y_level_pred) and np.isfinite(y_growth_pred):
        return (1.0 - w) * float(y_level_pred) + w * float(y_growth_pred)
    if np.isfinite(y_level_pred):
        return float(y_level_pred)
    if np.isfinite(y_growth_pred):
        return float(y_growth_pred)
    return float("nan")

# =========================================================
# Features por año + pesos
# =========================================================
def exp_weights_for_years(years: np.ndarray, last_obs_year: int, half_life: float = 4.0, boost_recent_years: int = 10) -> np.ndarray:
    years = np.asarray(years, dtype="int64")
    k = np.log(2.0) / max(half_life, 1e-9)
    w = np.exp(k * (years - int(last_obs_year)))
    if boost_recent_years and boost_recent_years > 0:
        cutoff = int(last_obs_year) - (boost_recent_years - 1)
        w *= np.where(years >= cutoff, 1.5, 1.0)
    w = w / (np.max(w) + 1e-12)
    return w.astype(float)

def build_features_year_axis(years: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = np.asarray(years, dtype="int64")
    vals = np.asarray(vals, dtype="float64")
    order = np.argsort(years)
    years = years[order]
    vals = vals[order]

    rows = []
    for i in range(len(years)):
        lag1 = vals[i-1] if i-1 >= 0 else np.nan
        lag2 = vals[i-2] if i-2 >= 0 else np.nan
        prev3 = vals[max(0, i-3):i]
        rm3 = float(np.mean(prev3)) if len(prev3) else np.nan
        rs3 = float(np.std(prev3, ddof=1)) if len(prev3) > 1 else (0.0 if len(prev3) == 1 else np.nan)
        rows.append([float(years[i]), float(lag1), float(lag2), float(rm3), float(rs3), float(vals[i])])

    arr = np.asarray(rows, float)
    mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[mask]
    if arr.size == 0:
        return np.empty((0,5), float), np.empty((0,), float), np.empty((0,), int)

    X = arr[:, :5].astype(float)
    y = arr[:, 5].astype(float)
    yrs = arr[:, 0].astype(int)
    return X, y, yrs

def level_naive_last(y_month_year: pd.Series, target_year: int) -> float:
    y = y_month_year.dropna().astype(float).sort_index()
    y_tr = y[y.index.astype(int) < int(target_year)]
    if len(y_tr) == 0:
        return float("nan")
    return float(y_tr.iloc[-1])

def level_mean_k(y_month_year: pd.Series, target_year: int, k: int = 3) -> float:
    y = y_month_year.dropna().astype(float).sort_index()
    y_tr = y[y.index.astype(int) < int(target_year)]
    if len(y_tr) == 0:
        return float("nan")
    tail = y_tr.values[-min(k, len(y_tr)):]
    return float(np.nanmean(tail))

def growth_median_k(y_month_year: pd.Series, target_year: int, k: int = 5) -> float:
    y = y_month_year.dropna().astype(float).sort_index()
    g = growth_yoy(y).dropna()
    g_tr = g[g.index.astype(int) < int(target_year)]
    if len(g_tr) == 0:
        return float("nan")
    tail = g_tr.values[-min(k, len(g_tr)):]
    ghat = float(np.nanmedian(tail))
    return float(np.clip(ghat, -0.95, 2.0))

# =========================================================
# ETS / SARIMAX sobre serie anual del mes
# =========================================================
def fit_predict_ets_year(y_train: pd.Series, steps: int = 1) -> float:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y_train = y_train.dropna().astype(float)
    trend = "add" if len(y_train) >= 6 else None
    model = ExponentialSmoothing(
        y_train.values,
        trend=trend,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)
    yhat = model.forecast(steps)[-1]
    return float(yhat)

def fit_predict_sarimax_year(y_train: pd.Series, steps: int = 1) -> float:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y_train = y_train.dropna().astype(float)
    order = (1, 0, 0) if len(y_train) >= 6 else (0, 0, 0)
    mod = SARIMAX(
        y_train.values,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    yhat = mod.forecast(steps=steps)[-1]
    return float(yhat)

# =========================================================
# Tabulares: Linear/Ridge/MLP/HGB
# =========================================================
def fit_tabular_regressor(model_kind: str):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    if model_kind == "Linear":
        return Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    if model_kind == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=42))])
    if model_kind == "MLP":
        return Pipeline([("sc", StandardScaler()), ("m",
            MLPRegressor(hidden_layer_sizes=(64, 32),
                         activation="relu",
                         solver="adam",
                         alpha=1e-4,
                         learning_rate_init=1e-3,
                         max_iter=2000,
                         random_state=42)
        )])
    if model_kind == "HGB":
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=700,
            random_state=42
        )
    raise ValueError("model_kind no soportado")

def tabular_fit_level_model(y_month_year: pd.Series, model_kind: str, target_year: int):
    """
    Ajusta modelo de nivel (features) con años < target_year y devuelve (model, X1) listo para predecir.
    """
    y = y_month_year.dropna().astype(float).sort_index()
    y_tr = y[y.index.astype(int) < int(target_year)]
    if len(y_tr) < 5:
        raise ValueError(f"{model_kind}: pocos años para features (n={len(y_tr)})")

    years = y_tr.index.values.astype(int)
    vals = y_tr.values.astype(float)

    X, Y, yrs_f = build_features_year_axis(years, vals)
    if len(Y) < 3:
        raise ValueError(f"{model_kind}: pocos pares con features (n={len(Y)})")

    last_obs_year = int(np.max(yrs_f))
    w = exp_weights_for_years(yrs_f, last_obs_year, half_life=HALF_LIFE_YEARS, boost_recent_years=RECENT_YEARS)

    lag1 = vals[-1]
    lag2 = vals[-2] if len(vals) >= 2 else vals[-1]
    prev3 = vals[-min(3, len(vals)):]
    rm3 = float(np.mean(prev3))
    rs3 = float(np.std(prev3, ddof=1)) if len(prev3) > 1 else 0.0
    X1 = np.array([[float(target_year), float(lag1), float(lag2), float(rm3), float(rs3)]], float)

    model = fit_tabular_regressor(model_kind)

    if model_kind in ("Linear", "Ridge"):
        model.fit(X, Y, m__sample_weight=w)
    elif model_kind == "HGB":
        model.fit(X, Y, sample_weight=w)
    else:
        model.fit(X, Y)
    return model, X1

def tabular_predict_growth_lags(g_year: pd.Series, target_year: int, lags: int) -> float:
    g = g_year.dropna().astype(float).sort_index()
    g_tr = g[g.index.astype(int) < int(target_year)]
    min_pairs = 3
    min_years = lags + min_pairs
    if len(g_tr) < min_years:
        raise ValueError("growth: pocos datos")

    vals = g_tr.values.astype(float)
    X, Y = [], []
    for i in range(lags, len(vals)):
        X.append(vals[i-lags:i].copy())
        Y.append(vals[i])
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if len(X) < min_pairs:
        raise ValueError("growth: pocos pares")

    model = fit_tabular_regressor("Ridge")
    model.fit(X, Y)
    tail = vals[-lags:]
    ghat = float(model.predict(tail.reshape(1, -1))[0])
    return float(np.clip(ghat, -0.95, 2.0))

# =========================================================
# NN (train 1x por mes) + recursivo por años
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

def make_supervised_1d(y_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    y_scaled = np.asarray(y_scaled, float).reshape(-1)
    X, Y = [], []
    for i in range(lookback, len(y_scaled)):
        X.append(y_scaled[i - lookback:i].reshape(lookback, 1))
        Y.append([y_scaled[i]])
    return np.array(X, float), np.array(Y, float)

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
    x = layers.LSTM(32, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(16)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
    return m

def build_tcn(input_len: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    from tcn import TCN
    inp = layers.Input(shape=(input_len, 1))
    x = TCN(nb_filters=32, kernel_size=3, dilations=[1,2,4,8,16],
            padding="causal", dropout_rate=0.2, return_sequences=False,
            use_skip_connections=True)(inp)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mae")
    return m

def build_multitask_dl(input_len: int):
    tf = get_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
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
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    cb = [
        EarlyStopping(monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=max(3, NN_PATIENCE//2), factor=0.5, min_lr=1e-5),
    ]
    if not multitask:
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                  epochs=NN_EPOCHS, batch_size=NN_BATCH, shuffle=False, verbose=0, callbacks=cb)
    else:
        ytr_level = Ytr
        ytr_delta = (Ytr.reshape(-1) - Xtr[:, -1, 0]).reshape(-1, 1)
        yva_level = Yva
        yva_delta = (Yva.reshape(-1) - Xva[:, -1, 0]).reshape(-1, 1)
        model.fit(Xtr, {"y_level": ytr_level, "y_delta": ytr_delta},
                  validation_data=(Xva, {"y_level": yva_level, "y_delta": yva_delta}),
                  epochs=NN_EPOCHS, batch_size=NN_BATCH, shuffle=False, verbose=0, callbacks=cb)

def predict_keras_one_step(model, X):
    yhat = model.predict(X, verbose=0)
    if isinstance(yhat, list):
        return np.asarray(yhat[0]).reshape(-1)
    return np.asarray(yhat).reshape(-1)

def nn_train_month_model(y_month_year: pd.Series, lookback: int, kind: str):
    """
    Entrena 1 vez con todo el historial real (del mes). Devuelve (model, scaler, last_window_scaled)
    """
    y = y_month_year.dropna().astype(float).sort_index()
    if len(y) < (lookback + 6):
        raise ValueError("NN: pocos años")

    sc = fit_asinh_scaler(y.values)
    ys = transform_asinh(y.values, sc)

    X, Y = make_supervised_1d(ys, lookback)
    if len(X) < 6:
        raise ValueError("NN: pocos pares")

    ntr = len(X)
    nval = max(4, int(0.2 * ntr))
    if ntr <= nval:
        raise ValueError("NN: no alcanza para val")

    Xtrain, Ytrain = X[:-nval], Y[:-nval]
    Xval,   Yval   = X[-nval:], Y[-nval:]

    if kind == "LSTM":
        m = build_lstm(lookback)
        train_keras_model(m, Xtrain, Ytrain, Xval, Yval, multitask=False)
    elif kind == "TCN":
        m = build_tcn(lookback)
        train_keras_model(m, Xtrain, Ytrain, Xval, Yval, multitask=False)
    elif kind == "DL_MultiTask":
        m = build_multitask_dl(lookback)
        train_keras_model(m, Xtrain, Ytrain, Xval, Yval, multitask=True)
    else:
        raise ValueError("NN kind no soportado")

    last_window = ys[-lookback:].reshape(1, lookback, 1)
    return m, sc, last_window

def nn_forecast_years(m, sc, last_window, steps: int) -> np.ndarray:
    """
    Forecast recursivo 1-step, actualizando ventana con predicción.
    """
    w = last_window.astype(float).copy()
    preds_scaled = []
    for _ in range(int(steps)):
        yhat_s = float(predict_keras_one_step(m, w)[0])
        preds_scaled.append(yhat_s)
        # shift window
        w = np.concatenate([w[:, 1:, :], np.array([[[yhat_s]]])], axis=1)
    preds = inverse_asinh(np.asarray(preds_scaled, float), sc)
    return preds.reshape(-1)

# =========================================================
# Métricas / plots / export por código (igual)
# =========================================================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return float("nan")
    e = y_true[:m] - y_pred[:m]
    return float(np.sqrt(np.mean(e*e)))

def plot_all_models(codigo, y, train_end, test_idx, future_idx, preds_test, preds_future, out_png):
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Real", linewidth=2)
    plt.axvline(train_end, linestyle="--", linewidth=1)
    plt.text(train_end, np.nanmin(y.values), "  train_end", rotation=90, va="bottom")

    for name, p in preds_test.items():
        if p is None: 
            continue
        p = np.asarray(p, float).reshape(-1)
        if np.all(~np.isfinite(p)):
            continue
        m = min(len(test_idx), len(p))
        plt.plot(test_idx[:m], p[:m], label=f"{name} (test)")

    for name, p in preds_future.items():
        if p is None: 
            continue
        p = np.asarray(p, float).reshape(-1)
        if np.all(~np.isfinite(p)):
            continue
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (future)", linestyle=":")

    plt.title(f"Codigo {codigo} — Test {len(test_idx)}m + Forecast {len(future_idx)}m (mensual-indep robusto)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_forecast_only(codigo, y, train_end, future_idx, preds_future, out_png):
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label="Real", linewidth=2)

    if train_end is not None:
        plt.axvline(train_end, linestyle="--", linewidth=1)
        y_min = np.nanmin(y.values) if np.isfinite(np.nanmin(y.values)) else 0.0
        plt.text(train_end, y_min, "  train_end", rotation=90, va="bottom")

    drew_any = False
    for name, p in preds_future.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        if np.all(~np.isfinite(p)):
            continue
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (forecast)", linestyle="-")
        drew_any = True

    if not drew_any:
        plt.text(future_idx[0], np.nanmedian(y.values) if len(y) else 0.0,
                 "NO HAY PREDS (revisar datos/modelos)", fontsize=10)

    plt.title(f"Codigo {codigo} — Forecast {len(future_idx)}m (sin test) (mensual-indep robusto)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def export_csv_codigo(out_csv, y, test_idx, future_idx, preds_test, preds_future):
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
# Read dataset ancho -> long mensual por codigo
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

    month_cols = [c for c in df.columns if c != "codigo" and re.match(r"^[A-Za-zÁÉÍÓÚáéíóú]{3}-\d{2,4}$", str(c).strip())]
    if not month_cols:
        raise ValueError("No se detectaron columnas de mes tipo Mar-15/Mar-2015.")
    col_to_ts = {}
    for c in month_cols:
        try:
            col_to_ts[c] = parse_month_col(c)
        except Exception:
            continue
    month_cols = [c for c in month_cols if c in col_to_ts]
    if not month_cols:
        raise ValueError("Las columnas candidatas no pudieron parsearse a fechas.")

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
# Mensual-independiente: extracción por mes
# =========================================================
def extract_month_year_series(s: pd.Series, month: int) -> pd.Series:
    ss = s.dropna().astype(float)
    ss = ss[ss.index.month == int(month)]
    if ss.empty:
        return pd.Series(dtype=float)
    y = pd.Series(ss.values, index=ss.index.year.astype(int))
    y = y[~y.index.duplicated(keep="last")].sort_index()
    return y.astype(float)

# =========================================================
# Predicción 1 objetivo (usa rollback y blend) - mismo core
# =========================================================
def predict_one_month_target(y_month_year: pd.Series, target_year: int, model_name: str,
                             nn_cache: dict, nn_lookback: int = 5) -> float:
    y_month_year = y_month_year.dropna().astype(float).sort_index()
    if len(y_month_year) == 0:
        return float("nan")

    y_last = float("nan")
    if (target_year - 1) in y_month_year.index:
        y_last = float(y_month_year.loc[target_year - 1])

    g = growth_yoy(y_month_year)

    y_level_pred = float("nan")
    g_pred = float("nan")

    try:
        y_train = y_month_year[y_month_year.index.astype(int) < int(target_year)]
        if len(y_train) < 3:
            raise ValueError("muy pocos años")

        if model_name == "ETS":
            y_level_pred = fit_predict_ets_year(y_train, steps=1)

        elif model_name == "SARIMAX":
            y_level_pred = fit_predict_sarimax_year(y_train, steps=1)

        elif model_name in ("Linear", "Ridge", "MLP", "HGB"):
            mdl, X1 = tabular_fit_level_model(y_month_year, model_kind=model_name, target_year=target_year)
            y_level_pred = float(mdl.predict(X1)[0])
            try:
                g_pred = tabular_predict_growth_lags(g, target_year=target_year, lags=GROWTH_LAGS)
            except Exception:
                g_pred = growth_median_k(y_month_year, target_year, k=5)

        elif model_name in ("LSTM", "TCN", "DL_MultiTask"):
            # cache NN por (modelo, mes_id) -> (m,sc,last_window, last_year)
            key = ("NN_MONTH", model_name, int(y_month_year.index.max()))
            if key not in nn_cache:
                m, sc, last_w = nn_train_month_model(y_month_year, lookback=nn_lookback, kind=model_name)
                nn_cache[key] = (m, sc, last_w, int(y_month_year.index.max()))
            m, sc, last_w, last_real_year = nn_cache[key]
            # steps = target_year - (last_real_year)
            steps = int(target_year) - int(last_real_year)
            if steps <= 0:
                # estamos prediciendo dentro del historial (backtest): usa rollback (para ser coherente)
                raise ValueError("NN backtest -> rollback")
            yhat_arr = nn_forecast_years(m, sc, last_w, steps=steps)
            y_level_pred = float(yhat_arr[-1])
            try:
                g_pred = tabular_predict_growth_lags(g, target_year=target_year, lags=GROWTH_LAGS)
            except Exception:
                g_pred = growth_median_k(y_month_year, target_year, k=5)
        else:
            raise ValueError("modelo no soportado")

    except Exception:
        # ===== rollback =====
        try:
            y_train = y_month_year[y_month_year.index.astype(int) < int(target_year)]
            if len(y_train) >= 5:
                y_level_pred = fit_predict_ets_year(y_train, steps=1)
            else:
                raise ValueError("pocos")
        except Exception:
            y_level_pred = level_mean_k(y_month_year, target_year, k=3)
            if not np.isfinite(y_level_pred):
                y_level_pred = level_naive_last(y_month_year, target_year)

        g_pred = growth_median_k(y_month_year, target_year, k=5)

    if not np.isfinite(g_pred):
        g_pred = growth_median_k(y_month_year, target_year, k=5)

    return float(blend_level_and_growth(float(y_level_pred), float(g_pred), float(y_last)))

# =========================================================
# WALK-FORWARD por mes: backtest + forecast (SIN re-ejecutar por fecha)
# =========================================================
def predict_paths_walkforward_by_month(
    s_all: pd.Series,
    test_idx: pd.DatetimeIndex,
    future_idx: pd.DatetimeIndex,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regresa (pred_test_alineado_a_test_idx, pred_future_alineado_a_future_idx)
    usando estrategia por-mes (subserie anual del mes).

    Importante: el forecast futuro NO agrega predicciones al entrenamiento (para
    mantener equivalencia con tu script anterior).
    """
    nn_cache = {}  # cache NN por mes/modelo

    # construir mapa de fechas->pred
    pred_test_map = {}
    pred_future_map = {}

    # agrupar targets por mes
    all_test = list(pd.DatetimeIndex(test_idx))
    all_fut  = list(pd.DatetimeIndex(future_idx))

    for month in range(1, 13):
        y_my = extract_month_year_series(s_all, month=month)
        if y_my.empty:
            continue

        # ---- backtest targets de ese mes
        targets_test = [dt for dt in all_test if int(dt.month) == month]
        for dt in targets_test:
            ty = int(dt.year)
            yhat = predict_one_month_target(y_my, target_year=ty, model_name=model_name, nn_cache=nn_cache)
            pred_test_map[dt] = yhat

        # ---- future targets de ese mes
        targets_fut = [dt for dt in all_fut if int(dt.month) == month]
        # para acelerar, orden por año
        targets_fut = sorted(targets_fut, key=lambda d: (d.year, d.month))
        for dt in targets_fut:
            ty = int(dt.year)
            yhat = predict_one_month_target(y_my, target_year=ty, model_name=model_name, nn_cache=nn_cache)
            pred_future_map[dt] = yhat

    # ensamblar arrays en el mismo orden
    yhat_test = np.array([pred_test_map.get(dt, np.nan) for dt in pd.DatetimeIndex(test_idx)], float)
    yhat_fut  = np.array([pred_future_map.get(dt, np.nan) for dt in pd.DatetimeIndex(future_idx)], float)
    return yhat_test, yhat_fut

# =========================================================
# RUN por codigo (estructura igual)
# =========================================================
def run_for_codigo(codigo: str, s: pd.Series):
    s = s.dropna().astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    min_len = max(36, TEST_LEN + 24)
    if len(s) < min_len:
        print(f"[SKIP] {codigo}: muy corta global (n={len(s)}), min_len={min_len}")
        return None

    all_zero = is_all_zero_series(s)

    n = len(s)
    train_end_i = n - TEST_LEN
    train_end_date = s.index[train_end_i]

    y_train_full = s.iloc[:train_end_i]
    y_test  = s.iloc[train_end_i:]
    test_idx   = y_test.index
    future_idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(1), periods=H_FUTURE, freq="MS")

    preds_test: Dict[str, Optional[np.ndarray]] = {}
    preds_fut:  Dict[str, Optional[np.ndarray]] = {}
    scores: Dict[str, float] = {}

    model_list = []
    if RUN_ETS: model_list.append("ETS")
    if RUN_SARIMAX: model_list.append("SARIMAX")
    if RUN_LINEAR: model_list.append("Linear")
    if RUN_RIDGE: model_list.append("Ridge")
    if RUN_MLP: model_list.append("MLP")
    if RUN_HGB: model_list.append("HGB")
    if RUN_TCN: model_list.append("TCN")
    if RUN_LSTM: model_list.append("LSTM")
    if RUN_MULTITASK_DL: model_list.append("DL_MultiTask")

    if all_zero:
        for name in model_list:
            preds_test[name] = np.zeros(len(test_idx), float)
            preds_fut[name]  = np.zeros(len(future_idx), float)
            scores[name]     = 0.0
    else:
        # s_all para extraer subseries por mes (incluye train+test reales)
        s_all = s.copy()
        for name in model_list:
            try:
                yhat_test, yhat_fut = predict_paths_walkforward_by_month(
                    s_all=s_all,
                    test_idx=test_idx,
                    future_idx=future_idx,
                    model_name=name,
                )
                preds_test[name] = yhat_test
                preds_fut[name]  = yhat_fut
                scores[name] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {name} falló: {e}")
                preds_test[name] = None
                preds_fut[name]  = None

    # ranking
    scored = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e99)
    print(f"\n=== {codigo} (n={len(s)}, test={len(y_test)}) mensual-indep robusto WALKFWD ===")
    for name, r in scored:
        print(f"  {name:12s} RMSE_test = {r:,.4f}")

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

    plot_forecast_only(
        codigo=codigo,
        y=s,
        train_end=train_end_date,
        future_idx=future_idx,
        preds_future=preds_fut,
        out_png=out_png_fore
    )

    # Copiar forecast-only a carpeta extra
    try:
        ensure_dir(OUT_DIR_ONLY_FORECAST)
        if os.path.exists(out_png_fore):
            shutil.copy2(out_png_fore, os.path.join(OUT_DIR_ONLY_FORECAST, f"{codigo}.png"))
        else:
            print(f"[WARN] No existe {out_png_fore} (no se generó el png).")
    except Exception as e:
        print(f"[WARN] No se pudo copiar forecast-only de {codigo}: {e}")

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

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR_ONLY_FORECAST)

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
                    wide_by_model[model_name][codigo][global_month_cols[i]] = float(arr[i]) if np.isfinite(arr[i]) else np.nan

            if len(scores) > 0:
                best_model = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e99)[0][0]
                best_arr = preds_fut.get(best_model, None)
                if best_arr is not None:
                    best_arr = np.asarray(best_arr, float).reshape(-1)
                    m = min(len(best_arr), len(global_future_idx))
                    wide_best[codigo] = {}
                    for i in range(m):
                        wide_best[codigo][global_month_cols[i]] = float(best_arr[i]) if np.isfinite(best_arr[i]) else np.nan

            all_results.append({
                "codigo": res["codigo"],
                "future_idx": res["future_idx"],
                "preds_fut": res["preds_fut"],
            })

        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            continue

    # Excel wide (codigo,modelo,meses...)
    if all_results:
        out_xlsx = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_ALL_MODELS.xlsx")
        try:
            save_future_preds_wide_excel(out_xlsx, all_results)
            print(f"[OK] guardado Excel wide: {out_xlsx}")
        except Exception as e:
            print(f"[WARN] No se pudo guardar Excel wide: {e}")
    else:
        print("[WARN] all_results vacío: no hubo ningún código procesado exitosamente.")

    plt.close("all")

    # CSVs wide por modelo + BEST + BEST_GLOBAL
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

    plt.close("all")

if __name__ == "__main__":
    main()
