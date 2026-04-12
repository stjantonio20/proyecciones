#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediccion_mensual_ytd_walkforward.py

- Lee dataset ancho (codigo, Mar-15..Dic-25)
- Usa serie YTD (acumulada) pero MODELA incrementos mensuales (dif intra-año)
  y reconstruye YTD con cumsum mensual => Dic = suma meses.
- Backtest 2025:
    Train: Mar-2015..Dic-2024
    Test : Ene-2025..Ago-2025
    Val  : Sep-2025..Dic-2025
  Métrica: MAPE (y opcional RMSE)
- Forecast anual walk-forward:
    Entrena hasta Dic-2025 -> predice 2026
    Añade 2026 predicho como pseudo-real -> predice 2027
    Añade 2027 -> predice 2028
- Exporta:
    - plots por codigo
    - csv por codigo
    - wide 2025 por modelo + BEST
    - long powerbi 2025 (real + pred por modelo)
    - long powerbi 2026-2028 (pred por modelo)
    - long maestro (real hasta 2025 + pred 2025-2028)

Requisitos:
- pandas, numpy, matplotlib
- statsmodels
- scikit-learn
- tensorflow + keras-tcn (si activas DL)

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
# CONFIG
# =========================================================
CSV_PATH = "./dataset/Crediguate_actualizado_mensual.csv"
OUT_DIR  = "proyeccion_walkforward"
OUT_DIR_ONLY_FORECAST = os.path.join(OUT_DIR, "_solo_real_forecast")

ONLY_CODIGO = None  # "709110" o None
ONLY_CODIGO = "601101"  # "709110" o None

# Rango base requerido
START_DATE = pd.Timestamp("2015-03-01")  # Mar-15
END_REAL   = pd.Timestamp("2025-12-01")  # Dic-25 (fin de datos reales)

# Backtest 2025
EVAL_TRAIN_END = pd.Timestamp("2024-12-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END   = pd.Timestamp("2025-08-01")
VAL_START  = pd.Timestamp("2025-09-01")
VAL_END    = pd.Timestamp("2025-12-01")

# Walk-forward forecast
FUTURE_YEARS = [2026, 2027, 2028]  # predice años completos

# Export
WIDE_PREFIX = "proyeccion"
EXPORT_WIDE_2025 = True

# ========= banderas de modelos (activa/desactiva) =========
RUN_ETS          = True
RUN_SARIMAX      = True          # el "simple" que ya tenías (sobre eje-año en tu script original)
RUN_LINEAR       = True
RUN_RIDGE        = True
RUN_MLP          = False
RUN_HGB          = True

RUN_TCN          = False
RUN_LSTM         = False
RUN_MULTITASK_DL = False

# NUEVOS estacionales sobre serie mensual (incrementos) con s=12
RUN_ARIMA        = True
RUN_SARIMAX_12   = True
RUN_ETS_HW_12    = True

# NN params
NN_EPOCHS   = 200
NN_BATCH    = 16
NN_PATIENCE = 10

# Mezcla nivel+crecimiento (para tu lógica original mensual-independiente)
# Aquí la usamos solo en el submódulo por-mes si lo activas; para serie mensual
# normalmente no hace falta. La dejamos por compatibilidad.
GROWTH_BLEND_W = 0.35
EPS_DEN = 1e-9

# Features / growth
GROWTH_LAGS = 2
RECENT_YEARS = 10
HALF_LIFE_YEARS = 4.0

# =========================================================
# Utils: parse columnas tipo "Mar-15"
# =========================================================
MONTH_MAP = {
    # English
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    # Spanish
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

def fmt_month_cols(idx: pd.DatetimeIndex) -> List[str]:
    return [d.strftime("%b-%y") for d in pd.DatetimeIndex(idx)]

def is_all_zero_series(s: pd.Series) -> bool:
    v = pd.to_numeric(s.values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return True
    return np.nanmax(np.abs(v)) == 0.0

def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return float("nan")
    yt = y_true[:m]
    yp = y_pred[:m]
    den = np.maximum(np.abs(yt), 1e-9)
    return float(np.mean(np.abs(yt - yp) / den) * 100.0)

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return float("nan")
    e = y_true[:m] - y_pred[:m]
    return float(np.sqrt(np.mean(e*e)))

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

    month_cols = [
        c for c in df.columns
        if c != "codigo"
        and re.match(r"^[A-Za-zÁÉÍÓÚáéíóú]{3}-\d{2,4}$", str(c).strip())
    ]
    if not month_cols:
        raise ValueError("No se detectaron columnas tipo Mar-15/Mar-2015.")

    col_to_ts = {}
    for c in month_cols:
        try:
            col_to_ts[c] = parse_month_col(c)
        except Exception:
            continue
    month_cols = [c for c in month_cols if c in col_to_ts]
    if not month_cols:
        raise ValueError("No se pudieron parsear columnas de mes a fechas.")

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

def clip_to_required_range(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    s = s.loc[(s.index >= START_DATE) & (s.index <= END_REAL)]
    s = s.asfreq("MS")
    return s

# =========================================================
# YTD <-> Incrementos mensuales
# =========================================================
def ytd_to_increments(ytd: pd.Series) -> pd.Series:
    """
    Convierte YTD (acumulado) a incrementos mensuales:
      inc[Jan] = ytd[Jan]
      inc[m]   = ytd[m] - ytd[m-1] (mismo año)
    """
    ytd = ytd.sort_index().astype(float)
    inc = pd.Series(index=ytd.index, dtype=float)

    for dt in ytd.index:
        if not np.isfinite(ytd.loc[dt]):
            inc.loc[dt] = np.nan
            continue
        if dt.month == 1:
            inc.loc[dt] = float(ytd.loc[dt])
        else:
            prev = (dt - pd.offsets.MonthBegin(1))
            if prev in ytd.index and np.isfinite(ytd.loc[prev]):
                inc.loc[dt] = float(ytd.loc[dt]) - float(ytd.loc[prev])
            else:
                # si falta el mes previo, no podemos diferenciar
                inc.loc[dt] = np.nan
    return inc

def increments_to_ytd(inc: pd.Series) -> pd.Series:
    """
    Reconstruye YTD por año con suma acumulada dentro del mismo año.
    """
    inc = inc.sort_index().astype(float)
    ytd = pd.Series(index=inc.index, dtype=float)
    for yr, g in inc.groupby(inc.index.year):
        g = g.sort_index()
        ytd.loc[g.index] = np.cumsum(g.values)
    return ytd

def month_growth(a: float, b: float) -> float:
    # crecimiento MoM: (b-a)/|a|
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    den = max(abs(float(a)), 1e-9)
    return float((float(b) - float(a)) / den)

# =========================================================
# Modelos tabulares (con features y señal de crecimiento MoM)
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
                         max_iter=3000,
                         random_state=42)
        )])
    if model_kind == "HGB":
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=800,
            random_state=42
        )
    raise ValueError("model_kind no soportado")

def build_features_for_inc_prediction(
    train_inc: pd.Series,
    target_dt: pd.Timestamp,
    prev_month_inc_value: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Features para predecir el incremento de (target_dt).
    Entrenamiento usa TODOS los años previos para ESE MES, pero agrega señal del mes previo
    (del mismo año) como feature (cuando exista).

    X = [year, lag1_same_month, lag2_same_month, mean3_same_month, std3_same_month, prev_inc, prev_inc_growth]
    """
    m = int(target_dt.month)
    target_year = int(target_dt.year)

    # dataset: pares (año -> inc del mes m)
    ss = train_inc.dropna()
    ss = ss[ss.index.month == m]
    if ss.empty:
        return np.empty((0,7)), np.empty((0,)), np.empty((0,), int), np.empty((0,), float)

    y_by_year = pd.Series(ss.values, index=ss.index.year.astype(int)).sort_index()
    y_tr = y_by_year[y_by_year.index < target_year].dropna()
    if len(y_tr) < 5:
        return np.empty((0,7)), np.empty((0,)), np.empty((0,), int), np.empty((0,), float)

    years = y_tr.index.values.astype(int)
    vals  = y_tr.values.astype(float)

    rows = []
    for i in range(len(years)):
        yr = int(years[i])
        lag1 = vals[i-1] if i-1 >= 0 else np.nan
        lag2 = vals[i-2] if i-2 >= 0 else np.nan
        prev3 = vals[max(0, i-3):i]
        rm3 = float(np.mean(prev3)) if len(prev3) else np.nan
        rs3 = float(np.std(prev3, ddof=1)) if len(prev3) > 1 else (0.0 if len(prev3) == 1 else np.nan)

        # prev_inc del mismo año (mes m-1)
        if m == 1:
            prev_inc = np.nan
            prev_g = np.nan
        else:
            prev_dt = pd.Timestamp(year=yr, month=m-1, day=1)
            cur_dt  = pd.Timestamp(year=yr, month=m, day=1)
            if prev_dt in train_inc.index and cur_dt in train_inc.index and np.isfinite(train_inc.loc[prev_dt]) and np.isfinite(train_inc.loc[cur_dt]):
                prev_inc = float(train_inc.loc[prev_dt])
                prev_g = month_growth(float(train_inc.loc[prev_dt]), float(train_inc.loc[cur_dt]))
            else:
                prev_inc = np.nan
                prev_g = np.nan

        rows.append([float(yr), float(lag1), float(lag2), float(rm3), float(rs3), float(prev_inc), float(prev_g), float(vals[i])])

    arr = np.asarray(rows, float)
    mask = np.all(np.isfinite(arr[:, :7]), axis=1) & np.isfinite(arr[:, 7])
    arr = arr[mask]
    if arr.size == 0:
        return np.empty((0,7)), np.empty((0,)), np.empty((0,), int), np.empty((0,), float)

    X = arr[:, :7]
    Y = arr[:, 7]
    yrs = arr[:, 0].astype(int)

    last_obs_year = int(np.max(yrs))
    w = exp_weights_for_years(yrs, last_obs_year, half_life=HALF_LIFE_YEARS, boost_recent_years=RECENT_YEARS)

    # features para target
    # lag1/lag2/mean3/std3 del mismo mes (sobre historial por años)
    lag1_t = float(vals[-1])
    lag2_t = float(vals[-2]) if len(vals) >= 2 else float(vals[-1])
    prev3_t = vals[-min(3, len(vals)):]
    rm3_t = float(np.mean(prev3_t))
    rs3_t = float(np.std(prev3_t, ddof=1)) if len(prev3_t) > 1 else 0.0

    if m == 1:
        prev_inc_t = np.nan
        prev_g_t = np.nan
    else:
        prev_inc_t = float(prev_month_inc_value) if prev_month_inc_value is not None and np.isfinite(prev_month_inc_value) else np.nan
        # para crecimiento del mes previo al actual necesitamos inc(m-2)->inc(m-1)
        # lo manejamos fuera (opcional). Aquí lo dejamos NaN si no hay.
        prev_g_t = np.nan

    X1 = np.array([[float(target_year), lag1_t, lag2_t, rm3_t, rs3_t, prev_inc_t, prev_g_t]], float)
    return X, Y, yrs, w, X1

def predict_inc_tabular(train_inc: pd.Series, target_dt: pd.Timestamp, model_kind: str, prev_month_inc_value: Optional[float]) -> float:
    X, Y, yrs, w, X1 = build_features_for_inc_prediction(train_inc, target_dt, prev_month_inc_value)
    if len(Y) < 3:
        raise ValueError(f"{model_kind}: pocos datos tabulares para {target_dt:%Y-%m}")
    model = fit_tabular_regressor(model_kind)
    if model_kind in ("Linear", "Ridge"):
        model.fit(X, Y, m__sample_weight=w)
    elif model_kind == "HGB":
        model.fit(X, Y, sample_weight=w)
    else:
        model.fit(X, Y)
    return float(model.predict(X1)[0])

# =========================================================
# Modelos estacionales sobre serie mensual (incrementos)
# =========================================================
def fit_forecast_arima(train: pd.Series, steps: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    y = train.dropna().astype(float)
    # orden simple robusto; si falla, cae a fallback en caller
    mod = ARIMA(y.values, order=(1, 1, 1)).fit()
    fc = mod.forecast(steps=steps)
    return np.asarray(fc, float)

def fit_forecast_sarimax12(train: pd.Series, steps: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = train.dropna().astype(float)
    mod = SARIMAX(
        y.values,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    fc = mod.forecast(steps=steps)
    return np.asarray(fc, float)

def fit_forecast_ets_hw12(train: pd.Series, steps: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = train.dropna().astype(float)
    # HW aditivo; si tu serie tiene negativos fuertes, a veces conviene trend=None
    mod = ExponentialSmoothing(
        y.values,
        trend="add" if len(y) >= 24 else None,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)
    fc = mod.forecast(steps)
    return np.asarray(fc, float)

# =========================================================
# Keras (opcional, sobre serie mensual incrementos)
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
    try:
        from tcn import TCN
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
        from tcn import TCN

    inp = layers.Input(shape=(input_len, 1))
    x = TCN(
        nb_filters=32,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16],
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

def forecast_nn_monthly(train: pd.Series, steps: int, kind: str, lookback: int = 24) -> np.ndarray:
    y = train.dropna().astype(float)
    if len(y) < lookback + 12:
        raise ValueError(f"{kind}: muy corta (n={len(y)})")

    sc = fit_asinh_scaler(y.values)
    ys = transform_asinh(y.values, sc)

    X, Y = make_supervised_1d(ys, lookback)
    if len(X) < 20:
        raise ValueError(f"{kind}: pocos pares (n={len(X)})")

    ntr = len(X)
    nval = max(12, int(0.2 * ntr))
    if ntr <= nval:
        raise ValueError(f"{kind}: no alcanza para val")

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

    # forecast recursivo
    preds = []
    hist = ys.copy()
    for _ in range(steps):
        x = hist[-lookback:].reshape(1, lookback, 1)
        yhat_s = float(predict_keras_one_step(m, x)[0])
        preds.append(yhat_s)
        hist = np.concatenate([hist, [yhat_s]])
    preds = inverse_asinh(np.asarray(preds), sc)
    return np.asarray(preds, float)

# =========================================================
# Forecast anual por modelo, trabajando incrementos -> YTD
# =========================================================
def year_months(year: int) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(year=year, month=1, day=1), periods=12, freq="MS")

def model_forecast_year_increments(
    model_name: str,
    inc_hist: pd.Series,
    target_year: int,
) -> pd.Series:
    """
    Devuelve incrementos predichos para Ene..Dic del target_year.
    Para tabulares usa predicción secuencial por mes para poder usar prev_inc.
    Para estacionales usa forecast directo 12 pasos sobre serie mensual.
    """
    idx = year_months(target_year)

    # fallback naive: repetir último incremento observado
    def naive_fc():
        last = float(inc_hist.dropna().iloc[-1]) if len(inc_hist.dropna()) else 0.0
        return pd.Series([last]*12, index=idx, dtype=float)

    if inc_hist.dropna().empty:
        return pd.Series([0.0]*12, index=idx, dtype=float)

    # --- modelos estacionales (serie mensual completa) ---
    if model_name == "ARIMA":
        try:
            fc = fit_forecast_arima(inc_hist, steps=12)
            return pd.Series(fc, index=idx, dtype=float)
        except Exception:
            return naive_fc()

    if model_name == "SARIMAX_12":
        try:
            fc = fit_forecast_sarimax12(inc_hist, steps=12)
            return pd.Series(fc, index=idx, dtype=float)
        except Exception:
            return naive_fc()

    if model_name == "ETS_HW_12":
        try:
            fc = fit_forecast_ets_hw12(inc_hist, steps=12)
            return pd.Series(fc, index=idx, dtype=float)
        except Exception:
            return naive_fc()

    # --- DL (serie mensual completa) ---
    if model_name in ("LSTM_M", "TCN_M", "DL_MultiTask_M"):
        try:
            kind = "LSTM" if model_name == "LSTM_M" else ("TCN" if model_name == "TCN_M" else "DL_MultiTask")
            fc = forecast_nn_monthly(inc_hist, steps=12, kind=kind, lookback=24)
            return pd.Series(fc, index=idx, dtype=float)
        except Exception:
            return naive_fc()

    # --- tabulares por mes (usa patrón por-mes a través de años + prev_inc) ---
    if model_name in ("Linear", "Ridge", "MLP", "HGB"):
        preds = []
        prev_inc = None
        for dt in idx:
            try:
                yhat = predict_inc_tabular(inc_hist, dt, model_kind=model_name, prev_month_inc_value=prev_inc)
            except Exception:
                # fallback: promedio de últimos 3 incrementos del mismo mes (si existe)
                m = dt.month
                ss = inc_hist.dropna()
                ss = ss[ss.index.month == m]
                if len(ss) >= 3:
                    yhat = float(np.mean(ss.values[-3:]))
                elif len(ss) >= 1:
                    yhat = float(ss.values[-1])
                else:
                    yhat = float(inc_hist.dropna().iloc[-1])
            preds.append(float(yhat))
            prev_inc = float(yhat)
        return pd.Series(preds, index=idx, dtype=float)

    # --- compatibilidad: ETS/SARIMAX "simples" (no estacional) sobre serie mensual ---
    if model_name == "ETS":
        # ETS sin estacionalidad sobre incrementos mensuales
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            y = inc_hist.dropna().astype(float)
            mod = ExponentialSmoothing(
                y.values,
                trend="add" if len(y) >= 24 else None,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = mod.forecast(12)
            return pd.Series(np.asarray(fc, float), index=idx)
        except Exception:
            return naive_fc()

    if model_name == "SARIMAX":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y = inc_hist.dropna().astype(float)
            mod = SARIMAX(
                y.values,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            fc = mod.forecast(12)
            return pd.Series(np.asarray(fc, float), index=idx)
        except Exception:
            return naive_fc()

    # default fallback
    return naive_fc()

def model_forecast_year_ytd(model_name: str, ytd_hist: pd.Series, target_year: int) -> pd.Series:
    """
    Toma historial YTD real (hasta cierto corte), lo pasa a incrementos,
    predice incrementos para el año target, y reconstruye YTD del año target.
    """
    inc_hist = ytd_to_increments(ytd_hist)
    inc_fc = model_forecast_year_increments(model_name, inc_hist, target_year)
    ytd_fc = increments_to_ytd(inc_fc)
    return ytd_fc

# =========================================================
# Plots / exports
# =========================================================
def plot_all_models_ytd(
    codigo: str,
    ytd_real: pd.Series,
    train_end: pd.Timestamp,
    preds_2025_by_model: Dict[str, pd.Series],
    out_png: str,
    mape_test: Dict[str, float],
    mape_val: Dict[str, float],
):
    plt.figure(figsize=(13, 5))
    plt.plot(ytd_real.index, ytd_real.values, label="Real (YTD)", linewidth=2)
    plt.axvline(train_end, linestyle="--", linewidth=1)
    plt.text(train_end, np.nanmin(ytd_real.values), "  train_end", rotation=90, va="bottom")

    for name, ser in preds_2025_by_model.items():
        if ser is None or ser.dropna().empty:
            continue
        label = f"{name} | MAPE test={mape_test.get(name, np.nan):.2f}% val={mape_val.get(name, np.nan):.2f}%"
        plt.plot(ser.index, ser.values, label=label, linestyle="-")

    plt.title(f"Codigo {codigo} — Backtest 2025 (YTD) (Train<=Dic-2024)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor YTD")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=7)
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

    drew_any = False
    for name, p in preds_future.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        if np.all(~np.isfinite(p)):
            continue
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name}", linestyle="-")
        drew_any = True

    if not drew_any:
        plt.text(future_idx[0], np.nanmedian(y.values) if len(y) else 0.0,
                 "NO HAY PREDS (revisar datos/modelos)", fontsize=10)

    plt.title(f"Codigo {codigo} — Forecast (YTD)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor YTD")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def export_csv_codigo_long(
    out_csv: str,
    ytd_real: pd.Series,
    preds_by_model: Dict[str, pd.Series],
    split_map: Dict[pd.Timestamp, str],
):
    """
    CSV largo por codigo:
      fecha, real, split, modelo, pred
    """
    rows = []
    for dt, realv in ytd_real.items():
        rows.append({"fecha": dt, "modelo": "REAL", "pred": np.nan, "real": float(realv) if np.isfinite(realv) else np.nan, "split": split_map.get(dt, "")})

    for m, ser in preds_by_model.items():
        if ser is None:
            continue
        for dt, pv in ser.items():
            rows.append({
                "fecha": dt,
                "modelo": str(m),
                "pred": float(pv) if np.isfinite(pv) else np.nan,
                "real": float(ytd_real.loc[dt]) if dt in ytd_real.index and np.isfinite(ytd_real.loc[dt]) else np.nan,
                "split": split_map.get(dt, "")
            })

    df = pd.DataFrame(rows)
    df["codigo"] = None  # lo llenamos afuera si quieres; aquí dejamos estructura
    df.to_csv(out_csv, index=False)

def export_wide_by_model_2025(out_dir: str, wide_prefix: str, preds_2025_by_model: Dict[str, Dict[str, pd.Series]]):
    """
    Genera CSV ancho por modelo:
      codigo, Jan-25, Feb-25, ... Dec-25
    preds_2025_by_model[model][codigo] = Series YTD 2025
    """
    idx = year_months(2025)
    month_cols = fmt_month_cols(idx)

    for model, by_code in preds_2025_by_model.items():
        rows = {}
        for codigo, ser in by_code.items():
            rows[codigo] = {}
            for dt in idx:
                col = dt.strftime("%b-%y")
                rows[codigo][col] = float(ser.loc[dt]) if ser is not None and dt in ser.index and np.isfinite(ser.loc[dt]) else np.nan

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "codigo"
        for c in month_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[month_cols]
        df.reset_index().to_csv(os.path.join(out_dir, f"{wide_prefix}_{model}_2025.csv"), index=False)

def export_powerbi_long(
    out_csv,
    codigo,
    ytd_real,
    preds_2025,
    preds_future
):
    rows = []

    # ===== REAL =====
    for dt, v in ytd_real.items():
        rows.append({
            "codigo": codigo,
            "fecha": dt,
            "tipo": "real",
            "modelo": "REAL",
            "valor": float(v) if np.isfinite(v) else None
        })

    # ===== TEST + VAL (2025) =====
    for model, ser in preds_2025.items():
        if ser is None:
            continue
        for dt, v in ser.items():
            rows.append({
                "codigo": codigo,
                "fecha": dt,
                "tipo": "pred_2025",
                "modelo": model,
                "valor": float(v) if np.isfinite(v) else None
            })

    # ===== FUTURO (2026–2028) =====
    for model, ser in preds_future.items():
        if ser is None:
            continue
        for dt, v in ser.items():
            rows.append({
                "codigo": codigo,
                "fecha": dt,
                "tipo": "pred_future",
                "modelo": model,
                "valor": float(v) if np.isfinite(v) else None
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# =========================================================
# RUN por codigo
# =========================================================
def build_model_list() -> List[str]:
    model_list = []
    if RUN_ETS: model_list.append("ETS")
    if RUN_SARIMAX: model_list.append("SARIMAX")
    if RUN_LINEAR: model_list.append("Linear")
    if RUN_RIDGE: model_list.append("Ridge")
    if RUN_MLP: model_list.append("MLP")
    if RUN_HGB: model_list.append("HGB")

    if RUN_ARIMA: model_list.append("ARIMA")
    if RUN_SARIMAX_12: model_list.append("SARIMAX_12")
    if RUN_ETS_HW_12: model_list.append("ETS_HW_12")

    if RUN_LSTM: model_list.append("LSTM_M")
    if RUN_TCN: model_list.append("TCN_M")
    if RUN_MULTITASK_DL: model_list.append("DL_MultiTask_M")

    return model_list

def run_for_codigo(codigo: str, ytd: pd.Series):
    ytd = ytd.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    ytd = clip_to_required_range(ytd)

    if ytd.dropna().empty:
        print(f"[SKIP] {codigo}: serie vacía tras recorte")
        return None

    if is_all_zero_series(ytd):
        print(f"[SKIP] {codigo}: serie toda cero")
        return None

    # ========= definimos splits 2025 =========
    train_end = EVAL_TRAIN_END
    test_idx = pd.date_range(TEST_START, TEST_END, freq="MS")
    val_idx  = pd.date_range(VAL_START, VAL_END, freq="MS")
    idx_2025 = year_months(2025)

    # split map para export
    split_map = {}
    for dt in ytd.index:
        if dt <= train_end:
            split_map[dt] = "train"
    for dt in test_idx:
        split_map[dt] = "test"
    for dt in val_idx:
        split_map[dt] = "val"

    # ========= backtest 2025 (train hasta dic-2024) =========
    ytd_train = ytd.loc[ytd.index <= train_end].copy()
    ytd_2025_real = ytd.loc[(ytd.index >= pd.Timestamp("2025-01-01")) & (ytd.index <= pd.Timestamp("2025-12-01"))].copy()

    model_list = build_model_list()

    preds_2025 = {}
    mape_test = {}
    mape_val = {}

    for m in model_list:
        try:
            ytd_fc_2025 = model_forecast_year_ytd(m, ytd_train, 2025)
            # asegurar index completo 2025
            ytd_fc_2025 = ytd_fc_2025.reindex(idx_2025)

            preds_2025[m] = ytd_fc_2025

            # métricas
            yt_test = ytd.loc[test_idx].values if all(dt in ytd.index for dt in test_idx) else ytd_2025_real.reindex(test_idx).values
            yp_test = ytd_fc_2025.reindex(test_idx).values
            mape_test[m] = safe_mape(yt_test, yp_test)

            yt_val = ytd.loc[val_idx].values if all(dt in ytd.index for dt in val_idx) else ytd_2025_real.reindex(val_idx).values
            yp_val = ytd_fc_2025.reindex(val_idx).values
            mape_val[m] = safe_mape(yt_val, yp_val)

        except Exception as e:
            print(f"[WARN] {codigo} {m} falló backtest 2025: {e}")
            preds_2025[m] = None
            mape_test[m] = np.nan
            mape_val[m] = np.nan

    # ========= escoger BEST por MAPE test (promedio) =========
    valid = [(m, mape_test.get(m, np.nan)) for m in model_list if np.isfinite(mape_test.get(m, np.nan))]
    best_model = sorted(valid, key=lambda x: x[1])[0][0] if valid else None

    # ========= walk-forward 2026-2028 =========
    # historia base real hasta Dic-2025
    ytd_hist_for_future = ytd.loc[ytd.index <= END_REAL].copy()

    preds_future_years = {}  # model -> Series 2026-2028 (YTD concatenado por año)
    for m in model_list:
        try:
            ytd_hist_work = ytd_hist_for_future.copy()
            all_fc = []
            for yr in FUTURE_YEARS:
                yfc = model_forecast_year_ytd(m, ytd_hist_work, yr)  # YTD del año yr
                yfc = yfc.reindex(year_months(yr))
                all_fc.append(yfc)

                # “como si fuera real”: extendemos historia con el año predicho
                ytd_hist_work = pd.concat([ytd_hist_work, yfc]).sort_index()

            preds_future_years[m] = pd.concat(all_fc).sort_index()
        except Exception as e:
            print(f"[WARN] {codigo} {m} falló future 2026-2028: {e}")
            preds_future_years[m] = None

    # ========= outputs por-código =========
    out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
    ensure_dir(out_c_dir)

    # plot comparativo backtest 2025
    out_png_2025 = os.path.join(out_c_dir, f"plot_backtest_2025_{codigo}.png")
    plot_all_models_ytd(
        codigo=codigo,
        ytd_real=ytd,
        train_end=train_end,
        preds_2025_by_model={k: v for k, v in preds_2025.items() if v is not None},
        out_png=out_png_2025,
        mape_test=mape_test,
        mape_val=mape_val
    )

    # plot forecast only 2026-2028 (solo pred)
    future_idx = pd.date_range(pd.Timestamp("2026-01-01"), pd.Timestamp("2028-12-01"), freq="MS")
    preds_future_for_plot = {}
    for m in model_list:
        ser = preds_future_years.get(m, None)
        if ser is None:
            preds_future_for_plot[m] = None
        else:
            preds_future_for_plot[m] = ser.reindex(future_idx).values

    out_png_fore = os.path.join(out_c_dir, f"plot_forecast_2026_2028_{codigo}.png")
    plot_forecast_only(
        codigo=codigo,
        y=ytd,  # real hasta 2025
        train_end=END_REAL,
        future_idx=future_idx,
        preds_future=preds_future_for_plot,
        out_png=out_png_fore
    )

    export_powerbi_long(
        out_csv=os.path.join(out_c_dir, f"powerbi_long_{codigo}.csv"),
        codigo=codigo,
        ytd_real=ytd,
        preds_2025=preds_2025,
        preds_future=preds_future_years   # <--- ESTA
    )


    # Copiar forecast-only a carpeta extra
    try:
        ensure_dir(OUT_DIR_ONLY_FORECAST)
        if os.path.exists(out_png_fore):
            shutil.copy2(out_png_fore, os.path.join(OUT_DIR_ONLY_FORECAST, f"{codigo}.png"))
    except Exception as e:
        print(f"[WARN] No se pudo copiar forecast-only de {codigo}: {e}")

    # CSV largo por codigo (PowerBI-friendly) con real + 2025 preds + 2026-2028 preds
    # Armamos un maestro por codigo
    rows = []
    # real hasta 2025
    for dt, rv in ytd.items():
        rows.append({"codigo": codigo, "fecha": dt, "tipo": "real", "modelo": "REAL", "valor": float(rv), "split": split_map.get(dt, "")})

    # predicciones 2025 (backtest)
    for m, ser in preds_2025.items():
        if ser is None:
            continue
        for dt, pv in ser.items():
            rows.append({"codigo": codigo, "fecha": dt, "tipo": "pred_2025", "modelo": m, "valor": float(pv) if np.isfinite(pv) else np.nan, "split": split_map.get(dt, "")})

    # predicciones 2026-2028
    for m, ser in preds_future_years.items():
        if ser is None:
            continue
        for dt, pv in ser.items():
            rows.append({"codigo": codigo, "fecha": dt, "tipo": "pred_future", "modelo": m, "valor": float(pv) if np.isfinite(pv) else np.nan, "split": "future"})

    out_csv_long = os.path.join(out_c_dir, f"powerbi_long_{codigo}.csv")
    pd.DataFrame(rows).to_csv(out_csv_long, index=False)

    print(f"[OK] {codigo} plot_2025: {out_png_2025}")
    print(f"[OK] {codigo} plot_future: {out_png_fore}")
    print(f"[OK] {codigo} csv_long: {out_csv_long}")
    print(f"[INFO] {codigo} BEST_MODEL_2025_by_MAPEtest: {best_model}")

    return {
        "codigo": codigo,
        "preds_2025": preds_2025,
        "mape_test": mape_test,
        "mape_val": mape_val,
        "best_model": best_model,
        "preds_future": preds_future_years,
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

    model_list = build_model_list()

    # para exports globales
    # wide 2025 por modelo: model -> codigo -> Series
    wide_2025_by_model: Dict[str, Dict[str, pd.Series]] = {m: {} for m in model_list}
    wide_2025_best: Dict[str, pd.Series] = {}

    # long globales powerbi
    powerbi_2025_rows = []
    powerbi_future_rows = []
    master_rows = []

    for c in codigos:
        try:
            ytd = series_by_codigo(long, str(c))
            res = run_for_codigo(str(c), ytd)
            if res is None:
                continue

            codigo = res["codigo"]
            preds_2025 = res["preds_2025"]
            preds_future = res["preds_future"]
            best_model = res.get("best_model", None)

            # ---- acumular wide 2025 ----
            for m in model_list:
                ser = preds_2025.get(m, None)
                if ser is not None:
                    wide_2025_by_model[m][codigo] = ser

            if best_model is not None and preds_2025.get(best_model, None) is not None:
                wide_2025_best[codigo] = preds_2025[best_model]

            # ---- long powerbi 2025 ----
            idx_2025 = year_months(2025)
            ytd_real_cut = clip_to_required_range(ytd).reindex(idx_2025)
            for dt in idx_2025:
                rv = ytd_real_cut.loc[dt] if dt in ytd_real_cut.index else np.nan
                powerbi_2025_rows.append({"codigo": codigo, "fecha": dt, "modelo": "REAL", "valor": float(rv) if np.isfinite(rv) else np.nan, "tipo": "real_2025"})

            for m, ser in preds_2025.items():
                if ser is None:
                    continue
                ser = ser.reindex(idx_2025)
                for dt in idx_2025:
                    pv = ser.loc[dt]
                    powerbi_2025_rows.append({"codigo": codigo, "fecha": dt, "modelo": m, "valor": float(pv) if np.isfinite(pv) else np.nan, "tipo": "pred_2025"})

            # ---- long powerbi future 2026-2028 ----
            fut_idx = pd.date_range(pd.Timestamp("2026-01-01"), pd.Timestamp("2028-12-01"), freq="MS")
            for m, ser in preds_future.items():
                if ser is None:
                    continue
                ser = ser.reindex(fut_idx)
                for dt in fut_idx:
                    pv = ser.loc[dt]
                    powerbi_future_rows.append({"codigo": codigo, "fecha": dt, "modelo": m, "valor": float(pv) if np.isfinite(pv) else np.nan, "tipo": "pred_2026_2028"})

            # ---- maestro (real hasta 2025 + pred 2025-2028) ----
            # real
            ytd_real_all = clip_to_required_range(ytd)
            for dt, rv in ytd_real_all.items():
                master_rows.append({"codigo": codigo, "fecha": dt, "modelo": "REAL", "valor": float(rv) if np.isfinite(rv) else np.nan, "tipo": "real"})
            # pred 2025
            for m, ser in preds_2025.items():
                if ser is None:
                    continue
                for dt, pv in ser.items():
                    master_rows.append({"codigo": codigo, "fecha": dt, "modelo": m, "valor": float(pv) if np.isfinite(pv) else np.nan, "tipo": "pred_2025"})
            # pred future
            for m, ser in preds_future.items():
                if ser is None:
                    continue
                for dt, pv in ser.items():
                    master_rows.append({"codigo": codigo, "fecha": dt, "modelo": m, "valor": float(pv) if np.isfinite(pv) else np.nan, "tipo": "pred_2026_2028"})

        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            continue

    # ========= exports globales =========
    # wide 2025 por modelo
    if EXPORT_WIDE_2025:
        export_wide_by_model_2025(OUT_DIR, WIDE_PREFIX, wide_2025_by_model)
        print(f"[OK] wide 2025 por modelo guardado en {OUT_DIR}")

        # BEST 2025
        if wide_2025_best:
            idx = year_months(2025)
            month_cols = fmt_month_cols(idx)
            rows = {}
            for codigo, ser in wide_2025_best.items():
                rows[codigo] = {}
                for dt in idx:
                    col = dt.strftime("%b-%y")
                    rows[codigo][col] = float(ser.loc[dt]) if dt in ser.index and np.isfinite(ser.loc[dt]) else np.nan
            df = pd.DataFrame.from_dict(rows, orient="index")
            df.index.name = "codigo"
            for c in month_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[month_cols]
            df.reset_index().to_csv(os.path.join(OUT_DIR, f"{WIDE_PREFIX}_BEST_2025.csv"), index=False)
            print(f"[OK] wide BEST 2025: {os.path.join(OUT_DIR, f'{WIDE_PREFIX}_BEST_2025.csv')}")

    # long powerbi 2025
    if powerbi_2025_rows:
        outp = os.path.join(OUT_DIR, "powerbi_2025_long.csv")
        pd.DataFrame(powerbi_2025_rows).to_csv(outp, index=False)
        print(f"[OK] powerbi 2025 long: {outp}")

    # long powerbi future 2026-2028
    if powerbi_future_rows:
        outp = os.path.join(OUT_DIR, "powerbi_2026_2028_long.csv")
        pd.DataFrame(powerbi_future_rows).to_csv(outp, index=False)
        print(f"[OK] powerbi 2026-2028 long: {outp}")

    # maestro
    if master_rows:
        outp = os.path.join(OUT_DIR, "powerbi_master_real_2025_pred_2025_2028_long.csv")
        pd.DataFrame(master_rows).to_csv(outp, index=False)
        print(f"[OK] powerbi master long: {outp}")

    plt.close("all")

if __name__ == "__main__":
    main()
