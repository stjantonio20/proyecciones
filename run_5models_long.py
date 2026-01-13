#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== FORZAR CPU + desactivar XLA ======
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # no usar GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = ""                        # evita flags raros heredados

warnings.filterwarnings("ignore")
import random
random.seed(24)
np.random.seed(24)
# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "Crediguate_rampa_dia_mes.csv"     # largo: codigo,fecha,valor
OUT_DIR  = "outputs_long_full"

H_FUTURE  = 12        # 12 meses
TEST_LEN  = 6         # test fijo 6 meses

#ONLY_CODIGO = "101101"  # None para todos
ONLY_CODIGO = None  # None para todos

# NNs
LOOKBACK_NN = 32
NN_EPOCHS   = 300
NN_BATCH    = 36
NN_PATIENCE = 10

# Tabulares
LAGS_TABULAR = 28

# switches
RUN_ETS          = True
RUN_SARIMAX      = True
RUN_TCN          = True
RUN_LSTM         = True
RUN_MULTITASK_DL = True

RUN_LINEAR       = True
RUN_RIDGE        = True
RUN_MLP          = True
RUN_HGB          = True   # HGB (log1p + lags=28)

# Export global (ancho)
EXPORT_WIDE_FUTURE = True
WIDE_PREFIX = "wide_future"

# --- Plot filter: no dibujar modelos con escala exagerada ---
PLOT_FILTER_EXTREME = True
PLOT_EXTREME_RATIO = 50.0   # si el forecast es > 50x la escala real, se oculta
PLOT_REF_Q = 0.95           # escala real = quantil 95% de |y|
PLOT_PRED_Q = 0.95          # escala pred = quantil 95% de |pred|

PLOT_FILTER_GROUP_OUTLIERS = True
PLOT_GROUP_RATIO = 20.0   # si un modelo es > 15x la mediana de escalas de preds, se oculta
PLOT_GROUP_MIN_MODELS = 6 # aplica filtro si hay al menos 4 modelos con forecast válido


# =========================================================
# Helpers
# =========================================================
def filter_models_by_group(
    preds: Dict[str, Optional[np.ndarray]],
    ratio: float = 15.0,
    p_q: float = 0.95,
    min_models: int = 4
) -> Dict[str, Optional[np.ndarray]]:
    """
    Oculta modelos cuyo forecast tiene una escala robusta muy superior
    a la del grupo (comparado contra la mediana de escalas).
    """
    scales = {}
    for name, p in preds.items():
        if p is None:
            continue
        s = _robust_scale(np.asarray(p, float), q=p_q)
        if np.isfinite(s) and s > 0:
            scales[name] = s

    if len(scales) < min_models:
        return preds

    med = float(np.median(list(scales.values())))
    if not np.isfinite(med) or med <= 0:
        return preds

    out = dict(preds)
    for name, s in scales.items():
        if s > ratio * med:
            out[name] = None
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return float("nan")
    e = y_true[:m] - y_pred[:m]
    return float(np.sqrt(np.mean(e * e)))

def is_all_zero_series(s: pd.Series) -> bool:
    v = pd.to_numeric(s.values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return True
    return np.nanmax(np.abs(v)) == 0.0

def month_features(idx: pd.DatetimeIndex) -> np.ndarray:
    m = idx.month.values.astype(float)
    sinm = np.sin(2*np.pi*m/12.0)
    cosm = np.cos(2*np.pi*m/12.0)
    return np.c_[sinm, cosm]

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


def _robust_scale(arr: np.ndarray, q: float = 0.95) -> float:
    arr = np.asarray(arr, float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.nanquantile(np.abs(arr), q))

def filter_models_by_scale(
    y: pd.Series,
    preds: Dict[str, Optional[np.ndarray]],
    ratio: float = 50.0,
    y_q: float = 0.95,
    p_q: float = 0.95
) -> Dict[str, Optional[np.ndarray]]:
    """
    Devuelve un dict igual pero con modelos extremos puestos como None
    (para que NO se grafiquen).
    """
    # escala real robusta
    y_scale = _robust_scale(y.values, q=y_q)

    # si la serie real es casi cero, no filtramos (evita división por 0)
    if not np.isfinite(y_scale) or y_scale <= 0:
        return preds

    out = dict(preds)
    for name, p in preds.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        p_scale = _robust_scale(p, q=p_q)

        # si el forecast tiene escala absurda vs real => no graficar
        if np.isfinite(p_scale) and p_scale > ratio * y_scale:
            out[name] = None
    return out


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
    return np.sinh(z) * sc.s


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


# =========================================================
# Modelos clásicos: ETS / SARIMAX
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
# Tabular: lags + estacionalidad
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
        model = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    elif model_kind == "Ridge":
        model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=42))])
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
# Keras: TCN / LSTM / MultiTask
# =========================================================
def get_tf():
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")   # por si CUDA_VISIBLE_DEVICES no bastó
    except Exception:
        pass
    return tf


def build_lstm(input_len: int):
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_len, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    #m.compile(optimizer="adam", loss="logcosh")
    import tensorflow as tf
    m.compile(optimizer="adam", loss=tf.keras.losses.LogCosh())
    return m

def build_tcn(input_len: int):
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
    lookback = len(last_window_scaled)
    preds = []
    for _ in range(h_future):
        x = np.array(buf[-lookback:], float).reshape(1, lookback, 1)
        yhat = model.predict(x, verbose=0)
        if isinstance(yhat, list):
            y_next = float(yhat[0].reshape(-1)[0])
        else:
            y_next = float(np.asarray(yhat).reshape(-1)[0])
        preds.append(y_next)
        buf.append(y_next)
    return np.array(preds, float)


# =========================================================
# Plotting
# =========================================================
def plot_all_models(
    codigo: str,
    agg: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    test_idx: pd.DatetimeIndex,
    future_idx: pd.DatetimeIndex,
    preds_test: Dict[str, Optional[np.ndarray]],
    preds_future: Dict[str, Optional[np.ndarray]],
    out_png: str
):
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label=f"Real mensual ({agg})", linewidth=2)

    plt.axvline(train_cut, linestyle="--", linewidth=1)
    y_min = np.nanmin(y.values) if np.isfinite(np.nanmin(y.values)) else 0.0
    plt.text(train_cut, y_min, "  train_end", rotation=90, va="bottom")
    pt = preds_test
    pf = preds_future
    if PLOT_FILTER_EXTREME:
        # para el test filtramos con respecto a la escala real también
        pt = filter_models_by_scale(
            y=y, preds=preds_test,
            ratio=PLOT_EXTREME_RATIO,
            y_q=PLOT_REF_Q,
            p_q=PLOT_PRED_Q
        )
        pf = filter_models_by_scale(
            y=y, preds=preds_future,
            ratio=PLOT_EXTREME_RATIO,
            y_q=PLOT_REF_Q,
            p_q=PLOT_PRED_Q
        )
    for name, p in pt.items():
        if p is None:
            continue
        m = min(len(test_idx), len(p))
        plt.plot(test_idx[:m], p[:m], label=f"{name} (test)")

    for name, p in pf.items():
        if p is None:
            continue
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (forecast)", linestyle=":")

    plt.title(f"Codigo {codigo} — {agg} — Test {len(test_idx)}m + Forecast {len(future_idx)}m")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_forecast_only(
    codigo: str,
    agg: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    future_idx: pd.DatetimeIndex,
    preds_future: Dict[str, Optional[np.ndarray]],
    out_png: str
):
    """Toda la serie real + SOLO forecast de todos los modelos."""
    plt.figure(figsize=(13, 5))
    plt.plot(y.index, y.values, label=f"Real mensual ({agg})", linewidth=2)

    if train_cut is not None:
        plt.axvline(train_cut, linestyle="--", linewidth=1)
        y_min = np.nanmin(y.values) if np.isfinite(np.nanmin(y.values)) else 0.0
        plt.text(train_cut, y_min, "  train_end", rotation=90, va="bottom")

        # Filtrar modelos con escala exagerada (solo para plot)
        # Filtrar modelos con escala exagerada (solo para plot)
    pf = preds_future

    if PLOT_FILTER_EXTREME:
        pf = filter_models_by_scale(
            y=y,
            preds=pf,
            ratio=PLOT_EXTREME_RATIO,
            y_q=PLOT_REF_Q,
            p_q=PLOT_PRED_Q
        )

    if PLOT_FILTER_GROUP_OUTLIERS:
        pf = filter_models_by_group(
            preds=pf,
            ratio=PLOT_GROUP_RATIO,
            p_q=PLOT_PRED_Q,
            min_models=PLOT_GROUP_MIN_MODELS
        )

    for name, p in pf.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (forecast)", linestyle="-")

    plt.title(f"Codigo {codigo} — {agg} — SOLO Forecast {len(future_idx)}m")
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
    preds_test: Dict[str, Optional[np.ndarray]],
    preds_future: Dict[str, Optional[np.ndarray]]
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
# Read LONG (muchos valores por día) + agregación mensual
# =========================================================
def read_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in ["codigo", "fecha", "valor"]:
        if col not in df.columns:
            raise ValueError(f"El CSV debe tener columna '{col}'.")

    df["codigo"] = df["codigo"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values(["codigo", "fecha"])
    return df

def monthly_series(df_long: pd.DataFrame, codigo: str, agg: str) -> pd.Series:
    g = df_long[df_long["codigo"] == codigo].copy()
    if g.empty:
        return pd.Series(dtype=float)

    s = pd.Series(g["valor"].values, index=pd.to_datetime(g["fecha"])).sort_index()
    if agg == "mean":
        m = s.resample("MS").mean()
    elif agg == "median":
        m = s.resample("MS").median()
    else:
        raise ValueError("agg debe ser 'mean' o 'median'")
    m = m.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return m


# =========================================================
# Perfil intra-mes (día + slot) para expandir forecast mensual
# =========================================================
def build_intramonth_profile(df_long: pd.DataFrame, codigo: str, rep: str, last_months: int = 12):
    """
    Devuelve:
      - template_days: lista de días-del-mes típicos (p.ej. 1..25)
      - template_times: lista de tiempos (HH:MM:SS) típicos (p.ej. 10)
      - weight_table: dict[(day,slot)] -> weight, con mean(weight)=1
    """
    g = df_long[df_long["codigo"] == codigo].copy()
    if g.empty:
        return [], [], {}

    g = g.dropna(subset=["fecha", "valor"]).copy()
    g["fecha"] = pd.to_datetime(g["fecha"])
    g["month"] = g["fecha"].dt.to_period("M").dt.to_timestamp("MS")

    # tomar últimos N meses disponibles
    months = sorted(g["month"].unique())
    if len(months) > last_months:
        months = months[-last_months:]
    g = g[g["month"].isin(months)].copy()

    # day-of-month y time-of-day
    g["day"] = g["fecha"].dt.day.astype(int)
    g["time"] = g["fecha"].dt.strftime("%H:%M:%S")

    # ordenar dentro de cada (month, day) para asignar slot 0..n-1
    g = g.sort_values(["month", "day", "fecha"])
    g["slot"] = g.groupby(["month", "day"]).cumcount()

    # template_times: los slots más comunes por (time)
    # tomamos los 10 horarios más frecuentes globalmente
    time_counts = g["time"].value_counts()
    template_times = time_counts.head(10).index.tolist()

    # template_days: los días más frecuentes (típicamente 25)
    day_counts = g["day"].value_counts()
    template_days = sorted(day_counts.head(31).index.tolist())  # hasta 31 por si acaso

    # Para construir pesos: necesitamos mes->representante (mean/median)
    month_rep = g.groupby("month")["valor"].agg(rep)
    g = g.join(month_rep.rename("rep_val"), on="month")

    # evitar división por 0
    g = g[np.isfinite(g["rep_val"]) & (np.abs(g["rep_val"]) > 0)].copy()
    if g.empty:
        return template_days, template_times, {}

    g["ratio"] = g["valor"] / g["rep_val"]

    # quedarnos con entradas que coincidan con template_times (opcional)
    g = g[g["time"].isin(template_times)].copy()

    # recomputar slot SOLO con los template_times ordenados por tiempo
    # map time->slot fijo 0..9 según orden lexicográfico
    t_sorted = sorted(template_times)
    time_to_slot = {t:i for i,t in enumerate(t_sorted)}
    g["slot_fix"] = g["time"].map(time_to_slot).astype(int)

    # peso por (day,slot)
    w = g.groupby(["day", "slot_fix"])["ratio"].mean()

    # normalizar para que mean(weight)=1 (así el promedio mensual “cuadra”)
    w_vals = w.values.astype(float)
    w_mean = float(np.mean(w_vals)) if len(w_vals) else 1.0
    if not np.isfinite(w_mean) or w_mean == 0:
        w_mean = 1.0
    w = w / w_mean

    weight_table = {(int(d), int(s)): float(val) for (d,s), val in w.items()}

    # usamos días típicos hasta 25 si existe
    if len(template_days) > 25:
        template_days = template_days[:25]

    return template_days, t_sorted, weight_table

def expand_monthly_forecast_to_intramonth(
    codigo: str,
    future_months: pd.DatetimeIndex,
    monthly_forecast: np.ndarray,
    template_days: List[int],
    template_times: List[str],
    weight_table: Dict[Tuple[int,int], float],
    rep_label: str
) -> pd.DataFrame:
    """
    Genera timestamps futuros del tipo:
      para cada mes futuro:
        para cada día en template_days (si existe en el mes):
          para cada time (10):
             valor = monthly_forecast[m] * weight(day,slot)  (default weight=1)
    """
    monthly_forecast = np.asarray(monthly_forecast, float).reshape(-1)
    rows = []
    for i, m0 in enumerate(pd.DatetimeIndex(future_months)):
        y_m = float(monthly_forecast[i]) if i < len(monthly_forecast) else np.nan
        if not np.isfinite(y_m):
            continue

        year = m0.year
        month = m0.month
        days_in_month = pd.Period(m0, freq="M").days_in_month

        for d in template_days:
            if d < 1 or d > days_in_month:
                continue
            for slot, t in enumerate(template_times):
                w = weight_table.get((int(d), int(slot)), 1.0)
                val = y_m * float(w)
                ts = pd.Timestamp(f"{year:04d}-{month:02d}-{d:02d} {t}")
                rows.append((codigo, rep_label, ts, year, month, d, slot, val))

    df = pd.DataFrame(rows, columns=["codigo","rep","fecha","year","month","day","slot","pred"])
    return df


# =========================================================
# RUN por codigo y por agregación (mean / median)
# =========================================================
def run_for_codigo_monthly(codigo: str, agg: str, y: pd.Series):
    y = y.dropna().astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    need = max(LOOKBACK_NN, LAGS_TABULAR) + TEST_LEN + 2
    if len(y) < need:
        print(f"[SKIP] {codigo} {agg}: muy corta (n={len(y)} < {need})")
        return None

    all_zero = is_all_zero_series(y)

    n = len(y)
    test_len = TEST_LEN
    train_end_i = n - test_len
    if train_end_i <= max(LOOKBACK_NN, LAGS_TABULAR) + 5:
        print(f"[SKIP] {codigo} {agg}: train insuficiente (n={n}, test={test_len})")
        return None

    y_train = y.iloc[:train_end_i]
    y_test  = y.iloc[train_end_i:]
    test_idx = y_test.index

    future_idx = pd.date_range(start=y.index[-1] + pd.offsets.MonthBegin(1), periods=H_FUTURE, freq="MS")
    train_cut = y.index[train_end_i]

    preds_test: Dict[str, Optional[np.ndarray]] = {}
    preds_fut:  Dict[str, Optional[np.ndarray]] = {}
    scores: Dict[str, float] = {}

    model_names_all = ["ETS","SARIMAX","TCN","LSTM","DL_MultiTask","Linear","Ridge","MLP","HGB"]

    if all_zero:
        for name in model_names_all:
            preds_test[name] = np.zeros(len(test_idx), float)
            preds_fut[name]  = np.zeros(len(future_idx), float)
            scores[name]     = 0.0
    else:
        # ETS
        if RUN_ETS:
            try:
                yhat_test, yhat_fut = fit_predict_ets(y_train, test_len=len(y_test), h_future=H_FUTURE)
                preds_test["ETS"] = yhat_test
                preds_fut["ETS"]  = yhat_fut
                scores["ETS"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} ETS falló: {e}")
                preds_test["ETS"] = None
                preds_fut["ETS"]  = None

        # SARIMAX
        if RUN_SARIMAX:
            try:
                yhat_test, yhat_fut = fit_predict_sarimax(y_train, test_len=len(y_test), h_future=H_FUTURE)
                preds_test["SARIMAX"] = yhat_test
                preds_fut["SARIMAX"]  = yhat_fut
                scores["SARIMAX"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} SARIMAX falló: {e}")
                preds_test["SARIMAX"] = None
                preds_fut["SARIMAX"]  = None

        # Tabulares
        if RUN_LINEAR:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    y, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="Linear", use_log1p=False
                )
                preds_test["Linear"] = yhat_test
                preds_fut["Linear"]  = yhat_fut
                scores["Linear"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} Linear falló: {e}")
                preds_test["Linear"] = None
                preds_fut["Linear"]  = None

        if RUN_RIDGE:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    y, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="Ridge", use_log1p=False
                )
                preds_test["Ridge"] = yhat_test
                preds_fut["Ridge"]  = yhat_fut
                scores["Ridge"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} Ridge falló: {e}")
                preds_test["Ridge"] = None
                preds_fut["Ridge"]  = None

        if RUN_MLP:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    y, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="MLP", use_log1p=False
                )
                preds_test["MLP"] = yhat_test
                preds_fut["MLP"]  = yhat_fut
                scores["MLP"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} MLP falló: {e}")
                preds_test["MLP"] = None
                preds_fut["MLP"]  = None

        if RUN_HGB:
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    y, train_end_i=train_end_i, test_len=len(y_test), h_future=H_FUTURE,
                    lags=LAGS_TABULAR, model_kind="HGB", use_log1p=True
                )
                preds_test["HGB"] = yhat_test
                preds_fut["HGB"]  = yhat_fut
                scores["HGB"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} HGB falló: {e}")
                preds_test["HGB"] = None
                preds_fut["HGB"]  = None

        # NNs
        sc = fit_asinh_scaler(y_train.values)
        y_scaled_all = transform_asinh(y.values, sc)

        X_all, Y_all = make_supervised(y_scaled_all, LOOKBACK_NN)
        target_idx = y.index[LOOKBACK_NN:]

        train_mask = target_idx < y.index[train_end_i]
        test_mask  = target_idx >= y.index[train_end_i]

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
                m = build_tcn(LOOKBACK_NN)
                train_keras_model(m, Xtrain, Ytrain, Xval, Yval, multitask=False)
                yhat_test_s = predict_keras_one_step(m, Xte)
                yhat_test = inverse_asinh(yhat_test_s, sc)
                yhat_fut_s = forecast_keras_autoregressive(m, last_window, H_FUTURE)
                yhat_fut = inverse_asinh(yhat_fut_s, sc)
                preds_test["TCN"] = yhat_test
                preds_fut["TCN"]  = yhat_fut
                scores["TCN"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} TCN falló: {e}")
                preds_test["TCN"] = None
                preds_fut["TCN"]  = None

        if RUN_LSTM:
            try:
                m = build_lstm(LOOKBACK_NN)
                train_keras_model(m, Xtrain, Ytrain, Xval, Yval, multitask=False)
                yhat_test_s = predict_keras_one_step(m, Xte)
                yhat_test = inverse_asinh(yhat_test_s, sc)
                yhat_fut_s = forecast_keras_autoregressive(m, last_window, H_FUTURE)
                yhat_fut = inverse_asinh(yhat_fut_s, sc)
                preds_test["LSTM"] = yhat_test
                preds_fut["LSTM"]  = yhat_fut
                scores["LSTM"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} LSTM falló: {e}")
                preds_test["LSTM"] = None
                preds_fut["LSTM"]  = None

        if RUN_MULTITASK_DL:
            try:
                m = build_multitask_dl(LOOKBACK_NN)
                train_keras_model(m, Xtrain, Ytrain, Xval, Yval, multitask=True)
                yhat_test_s = predict_keras_one_step(m, Xte)
                yhat_test = inverse_asinh(yhat_test_s, sc)
                yhat_fut_s = forecast_keras_autoregressive(m, last_window, H_FUTURE)
                yhat_fut = inverse_asinh(yhat_fut_s, sc)
                preds_test["DL_MultiTask"] = yhat_test
                preds_fut["DL_MultiTask"]  = yhat_fut
                scores["DL_MultiTask"] = rmse(y_test.values, yhat_test)
            except Exception as e:
                print(f"[WARN] {codigo} {agg} MultiTask DL falló: {e}")
                preds_test["DL_MultiTask"] = None
                preds_fut["DL_MultiTask"]  = None

    # ranking
    scored = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e99)
    print(f"\n=== {codigo} mensual({agg}) n={len(y)} test={len(y_test)} ===")
    for name, r in scored:
        print(f"  {name:14s} RMSE_test = {r:,.4f}")

    out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
    ensure_dir(out_c_dir)

    out_png = os.path.join(out_c_dir, f"plot_{codigo}_{agg}.png")
    out_png_fore = os.path.join(out_c_dir, f"plot_forecast_only_{codigo}_{agg}.png")
    out_csv = os.path.join(out_c_dir, f"pred_{codigo}_{agg}.csv")

    plot_all_models(
        codigo=codigo, agg=agg,
        y=y, train_cut=train_cut,
        test_idx=test_idx, future_idx=future_idx,
        preds_test=preds_test, preds_future=preds_fut,
        out_png=out_png
    )
    plot_forecast_only(
        codigo=codigo, agg=agg,
        y=y, train_cut=train_cut,
        future_idx=future_idx, preds_future=preds_fut,
        out_png=out_png_fore
    )
    export_csv_codigo(
        out_csv=out_csv,
        y=y, test_idx=test_idx, future_idx=future_idx,
        preds_test=preds_test, preds_future=preds_fut
    )

    print(f"[OK] guardado: {out_png}")
    print(f"[OK] guardado: {out_png_fore}")
    print(f"[OK] guardado: {out_csv}")

    return {
        "codigo": codigo,
        "agg": agg,
        "future_idx": future_idx,
        "preds_fut": preds_fut,
        "scores": scores,
        "y_month": y
    }


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    df_long = read_long(CSV_PATH)

    codigos = sorted(df_long["codigo"].unique().tolist())
    if ONLY_CODIGO is not None:
        codigos = [str(ONLY_CODIGO)]

    print(f"[INFO] codigos a procesar: {len(codigos)}")
    print(f"[INFO] outputs: {OUT_DIR}")
    print(f"[INFO] input: {CSV_PATH}")

    # acumuladores globales por agg
    global_pack: Dict[str, Dict[str, Any]] = {
        "mean": {"wide_by_model": {}, "wide_best": {}, "rmse_by_model": {}, "global_future_idx": None, "global_month_cols": None},
        "median": {"wide_by_model": {}, "wide_best": {}, "rmse_by_model": {}, "global_future_idx": None, "global_month_cols": None},
    }

    for c in codigos:
        for agg in ["mean", "median"]:
            try:
                y_month = monthly_series(df_long, str(c), agg=agg)
                if y_month.empty:
                    print(f"[SKIP] {c} {agg}: serie mensual vacía")
                    continue

                res = run_for_codigo_monthly(str(c), agg, y_month)
                if res is None:
                    continue

                future_idx = res["future_idx"]
                preds_fut  = res["preds_fut"]
                scores     = res.get("scores", {})
                codigo     = res["codigo"]

                pack = global_pack[agg]

                # guardar future idx global
                if pack["global_future_idx"] is None:
                    pack["global_future_idx"] = future_idx
                    pack["global_month_cols"] = fmt_month_cols(future_idx)

                month_cols = pack["global_month_cols"]

                # acumula RMSE por modelo
                for mname, r in scores.items():
                    if r is None or (isinstance(r, float) and (not np.isfinite(r))):
                        continue
                    pack["rmse_by_model"].setdefault(mname, []).append(float(r))

                # wide por modelo
                for model_name, arr in preds_fut.items():
                    if arr is None:
                        continue
                    arr = np.asarray(arr, float).reshape(-1)
                    m = min(len(arr), len(pack["global_future_idx"]))
                    pack["wide_by_model"].setdefault(model_name, {}).setdefault(codigo, {})
                    for i in range(m):
                        pack["wide_by_model"][model_name][codigo][month_cols[i]] = float(arr[i])

                # BEST por cuenta (según RMSE)
                if len(scores) > 0:
                    best_model = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e99)[0][0]
                    best_arr = preds_fut.get(best_model, None)
                    if best_arr is not None:
                        best_arr = np.asarray(best_arr, float).reshape(-1)
                        m = min(len(best_arr), len(pack["global_future_idx"]))
                        pack["wide_best"][codigo] = {}
                        for i in range(m):
                            pack["wide_best"][codigo][month_cols[i]] = float(best_arr[i])

                # ---- Expansión intra-mes (día + 10 valores) usando el BEST por cuenta ----
                # perfil basado en mean y en median (para que tengas ambos archivos)
                out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
                ensure_dir(out_c_dir)

                if len(scores) > 0 and best_arr is not None:
                    # perfil para mean
                    tdays, ttimes, wtbl = build_intramonth_profile(df_long, codigo, rep="mean", last_months=12)
                    df_exp = expand_monthly_forecast_to_intramonth(
                        codigo=codigo,
                        future_months=future_idx,
                        monthly_forecast=best_arr,
                        template_days=tdays,
                        template_times=ttimes,
                        weight_table=wtbl,
                        rep_label="profile_mean"
                    )
                    out_exp = os.path.join(out_c_dir, f"future_intramonth_mean_{codigo}_{agg}.csv")
                    df_exp.to_csv(out_exp, index=False)
                    print(f"[OK] guardado: {out_exp}")

                    # perfil para median
                    tdays, ttimes, wtbl = build_intramonth_profile(df_long, codigo, rep="median", last_months=12)
                    df_exp = expand_monthly_forecast_to_intramonth(
                        codigo=codigo,
                        future_months=future_idx,
                        monthly_forecast=best_arr,
                        template_days=tdays,
                        template_times=ttimes,
                        weight_table=wtbl,
                        rep_label="profile_median"
                    )
                    out_exp = os.path.join(out_c_dir, f"future_intramonth_median_{codigo}_{agg}.csv")
                    df_exp.to_csv(out_exp, index=False)
                    print(f"[OK] guardado: {out_exp}")

            except Exception as e:
                print(f"[ERROR] {c} {agg}: {e}")
                continue

    # ===== export ancho global por agg =====
    if EXPORT_WIDE_FUTURE:
        for agg in ["mean", "median"]:
            pack = global_pack[agg]
            if pack["global_future_idx"] is None:
                continue

            month_cols = pack["global_month_cols"]

            # 1) un CSV por modelo
            for model_name, rows in pack["wide_by_model"].items():
                out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_{agg}_{model_name}.csv")
                export_wide_future_csv(out_path, rows, month_cols)
                print(f"[OK] guardado ancho: {out_path}")

            # 2) BEST por cuenta
            if len(pack["wide_best"]) > 0:
                out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_{agg}_BEST.csv")
                export_wide_future_csv(out_path, pack["wide_best"], month_cols)
                print(f"[OK] guardado ancho: {out_path}")

            # 3) BEST_GLOBAL por MEAN RMSE
            best_mean_model = None
            best_mean_val = float("inf")
            for mname, vals in pack["rmse_by_model"].items():
                if len(vals) == 0:
                    continue
                v = float(np.mean(vals))
                if v < best_mean_val:
                    best_mean_val = v
                    best_mean_model = mname

            if best_mean_model is not None and best_mean_model in pack["wide_by_model"]:
                out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_{agg}_BEST_GLOBAL_MEAN.csv")
                export_wide_future_csv(out_path, pack["wide_by_model"][best_mean_model], month_cols)
                print(f"[OK] guardado ancho: {out_path}  (BEST_GLOBAL_MEAN={best_mean_model}, RMSE_mean={best_mean_val:,.4f})")

            # 4) BEST_GLOBAL por MEDIAN RMSE
            best_med_model = None
            best_med_val = float("inf")
            for mname, vals in pack["rmse_by_model"].items():
                if len(vals) == 0:
                    continue
                v = float(np.median(vals))
                if v < best_med_val:
                    best_med_val = v
                    best_med_model = mname

            if best_med_model is not None and best_med_model in pack["wide_by_model"]:
                out_path = os.path.join(OUT_DIR, f"{WIDE_PREFIX}_{agg}_BEST_GLOBAL_MEDIAN.csv")
                export_wide_future_csv(out_path, pack["wide_by_model"][best_med_model], month_cols)
                print(f"[OK] guardado ancho: {out_path}  (BEST_GLOBAL_MEDIAN={best_med_model}, RMSE_median={best_med_val:,.4f})")


if __name__ == "__main__":
    main()
