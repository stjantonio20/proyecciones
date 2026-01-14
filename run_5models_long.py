#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import hashlib
import gc
from typing import List


# =========================================================
# GPU config (TF)
# =========================================================
USE_GPU = True
# ====== FORZAR CPU/GPU + desactivar XLA ======
if USE_GPU:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # opcional (menos ruido / más determinista)

def tf_configure_gpu():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[TF] GPUs detectadas: {len(gpus)} (memory_growth=True)")
        else:
            print("[TF] No GPU detected (tf.config.list_physical_devices('GPU') == [])")
    except Exception as e:
        print(f"[TF] GPU config warning: {e}")



# configurar TF después de env vars
if USE_GPU:
    tf_configure_gpu()

warnings.filterwarnings("ignore")
import random
random.seed(24)
np.random.seed(24)

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "./dataset/CreNuevo_rampa_diario.csv"     # largo: codigo,fecha,valor
OUT_DIR  = "outputs_long_intraday"

ONLY_CODIGO = None # o None
#ONLY_CODIGO = ["101101","103101"]  # o None

# ---- Frecuencia base (tu dataset es cada 48min) ----
BASE_FREQ = "48min"      # o None para inferir
POINTS_PER_DAY_HINT = 30

# Horizonte futuro
FUTURE_DAYS = 90

# NNs
LOOKBACK_NN = 16 #210
NN_EPOCHS   = 80
NN_BATCH    = 20 #64
NN_PATIENCE = 10 #8

# Tabulares
LAGS_TABULAR = 210

# switches
RUN_ETS          = True
RUN_SARIMAX      = True
RUN_TCN          = True
RUN_LSTM         = True
RUN_MULTITASK_DL = True

RUN_LINEAR       = True
RUN_RIDGE        = True
RUN_LASSO        = True
RUN_MLP          = True
RUN_HGB          = True
RUN_LGBM         = True

# =========================================================
# NUEVO: ventana para ETS/SARIMAX si tardan
# =========================================================
# Si None => usa todo y_train
ETS_TRAIN_LAST_DAYS    = 360   # pon 90/180/365; o None
SARIMAX_TRAIN_LAST_DAYS = 180  # pon 90/180/365; o None

# =========================================================
# NUEVO: transform robusta tabular para negativos
# =========================================================
# Opciones: "none", "signed_log1p"
TABULAR_TARGET_TRANSFORM = "signed_log1p"

# plots
PLOT_FILTER_EXTREME = True
PLOT_EXTREME_RATIO = 50.0
PLOT_REF_Q = 0.95
PLOT_PRED_Q = 0.95

# split
SPLIT_MODE = "ratio"
TRAIN_RATIO = 0.97
MIN_TEST_POINTS = 300

# cache
CACHE_ENABLED = True
CACHE_DIR = os.path.join(OUT_DIR, "_cache_models")
CACHE_VERSION = "v2_intraday_2026-01-13_ETSwindow_negrobust"


# =========================================================
# Memory helper
# =========================================================
def cleanup_memory(tag: str = ""):
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    if tag:
        print(f"[GC] cleanup_memory: {tag}")


# =========================================================
# Helpers generales
# =========================================================
def clip_by_history(x: np.ndarray, y_hist: pd.Series, q_low=0.001, q_high=0.999, mult=3.0):
    """
    Recorta x a un rango razonable derivado de la historia.
    mult=3.0 => expande el rango para no recortar demasiado.
    """
    x = np.asarray(x, float).reshape(-1)

    yh = y_hist.values.astype(float)
    yh = yh[np.isfinite(yh)]
    if yh.size < 100:
        return x

    lo = np.nanquantile(yh, q_low)
    hi = np.nanquantile(yh, q_high)
    iqr = hi - lo
    if not np.isfinite(iqr) or iqr <= 0:
        return x

    lo2 = lo - mult * iqr
    hi2 = hi + mult * iqr
    return np.clip(x, lo2, hi2)

def filter_reasonable_models_from_future(df_fut: pd.DataFrame, y_hist: pd.Series, ratio=30.0):
    """
    Usa tu drop_outlier_models pero además recorta valores absurdos
    y construye una lista de columnas pred_* "buenas".
    """
    pred_cols = [c for c in df_fut.columns if c.startswith("pred_") and c != "pred_ENSEMBLE"]
    preds = {c.replace("pred_", ""): df_fut[c].values for c in pred_cols}

    # 1) elimina modelos con escala/picos absurdos
    keep = drop_outlier_models(
        y_ref=y_hist, preds=preds,
        ratio=ratio, q_ref=0.95, q_pred=0.995
    )
    keep_cols = [f"pred_{m}" for m,v in keep.items() if v is not None and f"pred_{m}" in df_fut.columns]

    # 2) CLIP (por si alguno se cuela “finito pero enorme”)
    for c in keep_cols:
        df_fut[c] = clip_by_history(df_fut[c].values, y_hist)

    return keep_cols

def add_robust_ensemble(df_fut: pd.DataFrame, keep_cols: list):
    """
    ENSEMBLE robusto: mediana (mejor que media cuando hay outliers).
    """
    if not keep_cols:
        return
    df_fut["pred_ENSEMBLE_MEDIAN"] = df_fut[keep_cols].median(axis=1)
    df_fut["pred_ENSEMBLE_MEAN"]   = df_fut[keep_cols].mean(axis=1)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def as_series(pred, idx: pd.DatetimeIndex) -> Optional[pd.Series]:
    if pred is None:
        return None
    arr = np.asarray(pred, float).reshape(-1)
    m = min(len(arr), len(idx))
    if m <= 0:
        return None
    return pd.Series(arr[:m], index=pd.DatetimeIndex(idx[:m]))

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    m = min(len(y_true), len(y_pred))
    if m == 0:
        return float("nan")
    e = y_true[:m] - y_pred[:m]
    return float(np.sqrt(np.mean(e * e)))

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
    y_scale = _robust_scale(y.values, q=y_q)
    if not np.isfinite(y_scale) or y_scale <= 0:
        return preds
    out = dict(preds)
    for name, p in preds.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        p_scale = _robust_scale(p, q=p_q)
        if np.isfinite(p_scale) and p_scale > ratio * y_scale:
            out[name] = None
    return out

def _hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def ensure_cache_dir(codigo: str) -> str:
    d = os.path.join(CACHE_DIR, f"codigo_{codigo}")
    ensure_dir(d)
    return d

def cache_path(codigo: str, model_name: str, params: dict, ext: str) -> str:
    h = _hash_dict(params)
    d = ensure_cache_dir(codigo)
    return os.path.join(d, f"{model_name}__{h}__{CACHE_VERSION}.{ext}")

def keras_cache_path(codigo: str, name: str, params: dict) -> str:
    return cache_path(codigo, f"KERAS_{name}", params, "keras")

def is_all_zero_series(s: pd.Series) -> bool:
    v = pd.to_numeric(s.values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return True
    return np.nanmax(np.abs(v)) == 0.0

def infer_base_freq(series_idx: pd.DatetimeIndex) -> str:
    if len(series_idx) < 5:
        return "48min"
    diffs = np.diff(series_idx.view("i8")) / 1e9
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return "48min"
    med = float(np.median(diffs))
    mins = max(1, int(round(med / 60.0)))
    return f"{mins}min"

def steps_per_day_from_freq(freq: str) -> int:
    mins = int(freq.replace("min",""))
    return int(round((24*60) / mins))

def crop_last_days(y: pd.Series, last_days: Optional[int], min_points: int) -> pd.Series:
    """Recorta y a los últimos N días, pero asegura mínimo min_points."""
    if last_days is None:
        return y
    if y.empty:
        return y
    end = y.index.max()
    start = end - pd.Timedelta(days=int(last_days))
    yc = y.loc[y.index >= start]
    if len(yc) < min_points:
        yc = y.iloc[-min_points:]
    return yc


# =========================================================
# Time features (sin/cos)
# =========================================================
def time_features(idx: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(idx)
    tod = (idx.hour * 60 + idx.minute).astype(float)
    dow = idx.dayofweek.astype(float)
    mon = idx.month.astype(float)

    tod_sin = np.sin(2*np.pi * tod / (24*60))
    tod_cos = np.cos(2*np.pi * tod / (24*60))
    dow_sin = np.sin(2*np.pi * dow / 7.0)
    dow_cos = np.cos(2*np.pi * dow / 7.0)
    mon_sin = np.sin(2*np.pi * mon / 12.0)
    mon_cos = np.cos(2*np.pi * mon / 12.0)

    return np.c_[tod_sin, tod_cos, dow_sin, dow_cos, mon_sin, mon_cos]


# =========================================================
# Tabular transforms robustos (negativos OK)
# =========================================================
def signed_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))

def inv_signed_log1p(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return np.sign(z) * np.expm1(np.abs(z))

def apply_tabular_transform(y: np.ndarray) -> np.ndarray:
    if TABULAR_TARGET_TRANSFORM == "none":
        return y
    if TABULAR_TARGET_TRANSFORM == "signed_log1p":
        return signed_log1p(y)
    raise ValueError(f"TABULAR_TARGET_TRANSFORM inválido: {TABULAR_TARGET_TRANSFORM}")

def invert_tabular_transform(z: np.ndarray) -> np.ndarray:
    if TABULAR_TARGET_TRANSFORM == "none":
        return z
    if TABULAR_TARGET_TRANSFORM == "signed_log1p":
        return inv_signed_log1p(z)
    raise ValueError(f"TABULAR_TARGET_TRANSFORM inválido: {TABULAR_TARGET_TRANSFORM}")


# =========================================================
# Scaling robusto (asinh) para NNs
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
# ETS / SARIMAX con ventana opcional
# =========================================================
def fit_predict_ets(
    codigo: str,
    y_train: pd.Series,
    test_len: int,
    h_future: int,
    seasonal_periods: int,
    base_freq: str,
    train_cut: pd.Timestamp
):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # NUEVO: recorte a últimos N días para acelerar/robustez
    min_points = max(5 * seasonal_periods, 1000)
    y_fit = crop_last_days(y_train, ETS_TRAIN_LAST_DAYS, min_points=min_points)

    seasonal = "add" if len(y_fit) >= 4 * seasonal_periods else None
    trend    = "add" if len(y_fit) >= 2 * seasonal_periods else None

    params = {
        "seasonal_periods": int(seasonal_periods),
        "trend": trend,
        "seasonal": seasonal,
        "train_cut": str(train_cut),
        "freq": base_freq,
        "len_fit": int(len(y_fit)),
        "last_days": ETS_TRAIN_LAST_DAYS,
    }
    path = cache_path(codigo, "ETS", params, "joblib")

    if CACHE_ENABLED and os.path.exists(path):
        model = joblib.load(path)
    else:
        model = ExponentialSmoothing(
            y_fit.values,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True)
        if CACHE_ENABLED:
            joblib.dump(model, path, compress=3)

    yhat_test = model.forecast(test_len)
    yhat_fut  = model.forecast(test_len + h_future)[-h_future:]
    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


def fit_predict_sarimax(
    codigo: str,
    y_train: pd.Series,
    test_len: int,
    h_future: int,
    seasonal_periods: int,
    base_freq: str,
    train_cut: pd.Timestamp
):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # NUEVO: recorte a últimos N días
    min_points = max(5 * seasonal_periods, 1500)
    y_fit = crop_last_days(y_train, SARIMAX_TRAIN_LAST_DAYS, min_points=min_points)

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, seasonal_periods) if len(y_fit) >= 4 * seasonal_periods else (0, 0, 0, 0)

    params = {
        "order": order,
        "seasonal_order": seasonal_order,
        "train_cut": str(train_cut),
        "freq": base_freq,
        "len_fit": int(len(y_fit)),
        "last_days": SARIMAX_TRAIN_LAST_DAYS,
    }
    path = cache_path(codigo, "SARIMAX", params, "joblib")

    if CACHE_ENABLED and os.path.exists(path):
        mod = joblib.load(path)
    else:
        mod = SARIMAX(
            y_fit.values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        if CACHE_ENABLED:
            joblib.dump(mod, path, compress=3)

    yhat_test = mod.forecast(steps=test_len)
    yhat_fut  = mod.forecast(steps=test_len + h_future)[-h_future:]
    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# Tabular: lags + time_features (robusto con negativos)
# =========================================================
def build_tabular_XY(y: pd.Series, lags: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    yv = y.values.astype(float)
    idx = y.index

    y_t = apply_tabular_transform(yv)

    X, Y, X_idx = [], [], []
    feats_time = time_features(idx)

    for t in range(lags, len(y_t)):
        l = y_t[t-lags:t]
        feats = np.r_[l, feats_time[t]]
        X.append(feats)
        Y.append(y_t[t])
        X_idx.append(idx[t])

    return np.array(X, float), np.array(Y, float), pd.DatetimeIndex(X_idx)

def tabular_forecast_autoreg(model, y_full: pd.Series, lags: int, future_idx: pd.DatetimeIndex) -> np.ndarray:
    yv = y_full.values.astype(float)
    buf = apply_tabular_transform(yv).tolist()

    preds_t = []
    for ts in future_idx:
        l = np.array(buf[-lags:], float)
        tf = time_features(pd.DatetimeIndex([ts])).ravel()
        feats = np.r_[l, tf].reshape(1, -1)
        y_next_t = float(model.predict(feats)[0])
        preds_t.append(y_next_t)
        buf.append(y_next_t)

    preds_t = np.asarray(preds_t, float)
    return invert_tabular_transform(preds_t)

def fit_predict_tabular_model(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    test_len: int,
    future_idx: pd.DatetimeIndex,
    lags: int,
    model_kind: str,
    base_freq: str,
) -> Tuple[np.ndarray, np.ndarray]:

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, Y, X_idx = build_tabular_XY(y, lags=lags)

    train_mask = (X_idx < train_cut)
    test_mask  = (X_idx >= train_cut)

    Xtr, Ytr = X[train_mask], Y[train_mask]
    Xte, Yte = X[test_mask],  Y[test_mask]
    Xte, Yte = Xte[:test_len], Yte[:test_len]

    if len(Xtr) < 200:
        raise ValueError(f"Tabular {model_kind}: muy pocos datos train (<200)")

    if model_kind == "Linear":
        model = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    elif model_kind == "Ridge":
        model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=42))])
    elif model_kind == "Lasso":
        model = Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=1e-3, random_state=42, max_iter=5000))])
    elif model_kind == "MLP":
        base = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=3000,
            random_state=42
        )
        model = Pipeline([("sc", StandardScaler()), ("m", base)])
    elif model_kind == "HGB":
        model = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=800,
            random_state=42
        )
    elif model_kind == "LGBM":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    else:
        raise ValueError(f"model_kind no soportado: {model_kind}")

    params = {
        "model_kind": model_kind,
        "lags": int(lags),
        "transform": TABULAR_TARGET_TRANSFORM,
        "train_cut": str(train_cut),
        "freq": base_freq,
        "time_feats": "tod+dow+mon_sincos",
        "cache_version": CACHE_VERSION,
    }
    path = cache_path(codigo, f"TAB_{model_kind}", params, "joblib")

    if CACHE_ENABLED and os.path.exists(path):
        model = joblib.load(path)
    else:
        model.fit(Xtr, Ytr)
        if CACHE_ENABLED:
            joblib.dump(model, path, compress=3)

    # pred test (en espacio transformado)
    yhat_test_t = model.predict(Xte)
    yhat_test = invert_tabular_transform(yhat_test_t)

    # future autoreg
    yhat_fut = tabular_forecast_autoreg(
        model, y, lags=lags, future_idx=future_idx
    )
    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# Keras models + cache
# =========================================================
def load_or_train_keras(
    codigo: str,
    model_name: str,
    build_fn,
    train_fn,
    Xtrain, Ytrain, Xval, Yval,
    params: dict
):
    kpath = keras_cache_path(codigo, model_name, params)

    import tensorflow as tf
    custom_objects = None

    if model_name == "TCN":
        try:
            from tcn import TCN
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
            from tcn import TCN
        if CACHE_ENABLED and os.path.exists(kpath):
            custom_objects = {"TCN": TCN}

    if CACHE_ENABLED and os.path.exists(kpath):
        try:
            m = tf.keras.models.load_model(kpath, compile=True, custom_objects=custom_objects)
            return m, True
        except Exception as e:
            print(f"[WARN] no pude cargar {model_name} desde cache, reentreno. motivo={e}")

    m = build_fn(params["lookback"])
    train_fn(
        m, Xtrain, Ytrain, Xval, Yval,
        patience=NN_PATIENCE, epochs=NN_EPOCHS, batch=NN_BATCH,
        multitask=(model_name == "DL_MultiTask")
    )

    if CACHE_ENABLED:
        ensure_dir(os.path.dirname(kpath))
        m.save(kpath)

    return m, False


def build_lstm(input_len: int):
    from tensorflow.keras import layers, models
    import tensorflow as tf
    inp = layers.Input(shape=(input_len, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss=tf.keras.losses.LogCosh())
    return m

def build_tcn(input_len: int):
    from tensorflow.keras import layers, models
    import tensorflow as tf  # faltaba
    from tcn import TCN

    inp = layers.Input(shape=(input_len, 1))
    x = TCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32, 64],
        padding="causal",
        dropout_rate=0.15, #0.2
        return_sequences=False,
        use_skip_connections=True,
    )(inp)

    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.Huber(delta=1.0)
    )
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

def train_keras_model(model, Xtr, Ytr, Xva, Yva, patience: int, epochs: int, batch: int, multitask=False):
    from tensorflow.keras.callbacks import EarlyStopping
    cb = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if not multitask:
        model.fit(
            Xtr, Ytr,
            validation_data=(Xva, Yva),
            epochs=epochs,
            batch_size=batch,
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
            epochs=epochs,
            batch_size=batch,
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
def plot_forecast_intraday(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    future_idx: pd.DatetimeIndex,
    preds_future: Dict[str, Optional[np.ndarray]],
    out_png: str,
    max_points_plot: int = 8000
):
    plt.figure(figsize=(13, 5))
    y_plot = y
    if len(y_plot) > max_points_plot:
        step = max(1, len(y_plot)//max_points_plot)
        y_plot = y_plot.iloc[::step]
    plt.plot(y_plot.index, y_plot.values, label="Real intradía", linewidth=1.5)
    plt.axvline(train_cut, linestyle="--", linewidth=1)

    pf = preds_future
    if PLOT_FILTER_EXTREME:
        pf = filter_models_by_scale(y=y, preds=pf, ratio=PLOT_EXTREME_RATIO, y_q=PLOT_REF_Q, p_q=PLOT_PRED_Q)

    for name, p in pf.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        fidx = future_idx
        if len(fidx) > max_points_plot:
            step = max(1, len(fidx)//max_points_plot)
            fidx = fidx[::step]
            p = p[::step]
        m = min(len(fidx), len(p))
        plt.plot(fidx[:m], p[:m], label=f"{name} (forecast)", linewidth=1)

    plt.title(f"Codigo {codigo} — Forecast intradía (pasos={len(future_idx)})")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_test_vs_pred_intraday(
    codigo: str,
    y_test: pd.Series,
    preds_test_series: Dict[str, Optional[pd.Series]],
    train_cut: pd.Timestamp,
    out_png: str,
    max_points_plot: int = 8000
):
    plt.figure(figsize=(13, 5))
    y_plot = y_test
    if len(y_plot) > max_points_plot:
        step = max(1, len(y_plot)//max_points_plot)
        y_plot = y_plot.iloc[::step]
    plt.plot(y_plot.index, y_plot.values, label="Real (TEST)", linewidth=2)
    plt.axvline(train_cut, linestyle="--", linewidth=1)

    pf = {name: (None if s is None else s.values) for name, s in preds_test_series.items()}
    if PLOT_FILTER_EXTREME:
        pf = filter_models_by_scale(y=y_test, preds=pf, ratio=PLOT_EXTREME_RATIO, y_q=PLOT_REF_Q, p_q=PLOT_PRED_Q)

    for name, s in preds_test_series.items():
        if s is None:
            continue
        if pf.get(name, None) is None:
            continue
        common_idx = y_test.index.intersection(s.index)
        if len(common_idx) == 0:
            continue
        sp = s.loc[common_idx]
        if len(common_idx) > max_points_plot:
            step = max(1, len(common_idx)//max_points_plot)
            sp = sp.loc[common_idx[::step]]
        plt.plot(sp.index, sp.values, label=f"{name} (pred test)", linewidth=1)

    plt.title(f"Codigo {codigo} — TEST vs PRED (intradía)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _downsample_series(s: pd.Series, max_points: int = 15000) -> pd.Series:
    s = s.dropna()
    if len(s) <= max_points:
        return s
    step = max(1, len(s) // max_points)
    return s.iloc[::step]

def plot_full_series_raw(
    codigo: str,
    y: pd.Series,
    out_png: str,
    max_points_plot: int = 20000
):
    """(1) Serie completa tal cual (sin preds)."""
    plt.figure(figsize=(13, 5))
    yp = _downsample_series(y, max_points_plot)
    plt.plot(yp.index, yp.values, linewidth=1.8, label="Real intradía")
    plt.title(f"Codigo {codigo} — Serie completa intradía (raw)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_full_plus_future_intraday_only(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    df_fut: pd.DataFrame,
    out_png: str,
    max_points_plot: int = 20000,
    plot_models: bool = True,
    plot_ensemble: bool = True,
):
    """
    (2) Serie completa + SOLO predicción intradía futura (sin test).
    Si plot_models=True pinta todas las pred_*; si no, solo ENSEMBLE.
    """
    plt.figure(figsize=(13, 5))
    yp = _downsample_series(y, max_points_plot)
    plt.plot(yp.index, yp.values, linewidth=1.8, label="Real intradía")
    plt.axvline(train_cut, linestyle="--", linewidth=1)

    cols = [c for c in df_fut.columns if c.startswith("pred_")]
    if not plot_models:
        cols = ["pred_ENSEMBLE"] if "pred_ENSEMBLE" in df_fut.columns else []

    if not plot_ensemble:
        cols = [c for c in cols if c != "pred_ENSEMBLE"]

    for c in cols:
        sp = df_fut[c].dropna()
        if sp.empty:
            continue
        sp = _downsample_series(sp, max_points_plot)
        plt.plot(sp.index, sp.values, linewidth=1.0, label=c)

    plt.title(f"Codigo {codigo} — Serie completa + forecast intradía (solo futuro)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _series_mode_rounded(x: pd.Series, mode_round: int = 2) -> float:
    """Modo robusto con redondeo para floats."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    xr = x.round(mode_round)
    vc = xr.value_counts()
    if vc.empty:
        return np.nan
    return float(vc.idxmax())

def plot_daily_median_and_monthly_mode_from_future(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    df_fut: pd.DataFrame,
    out_png: str,
    out_csv_daily: str,
    out_csv_monthly: str,
    use_col: str = "pred_ENSEMBLE",
    mode_round: int = 2
):
    """
    (3) Guarda:
      - mediana diaria de las predicciones (futuro)
      - modo (valor más frecuente) mensual de las predicciones (futuro)
    y lo grafica junto con la serie original (agregada a diario/mensual para comparar).
    """
    if use_col not in df_fut.columns:
        return

    pred = df_fut[use_col].dropna()
    if pred.empty:
        return

    # daily median (future)
    pred_daily_median = pred.resample("D").median()

    # monthly mode (future)
    pred_monthly_mode = pred.resample("MS").apply(lambda s: _series_mode_rounded(s, mode_round=mode_round))

    # guardar csv
    pred_daily_median.rename("pred_daily_median").to_frame().reset_index().rename(columns={"index": "fecha"}).to_csv(out_csv_daily, index=False)
    pred_monthly_mode.rename("pred_monthly_mode").to_frame().reset_index().rename(columns={"index": "mes"}).to_csv(out_csv_monthly, index=False)

    # Para comparar con real, agregamos real a diario (median) y mensual (mode aproximado)
    y_daily = y.resample("D").median()
    y_month = y.resample("MS").apply(lambda s: _series_mode_rounded(s, mode_round=mode_round))

    plt.figure(figsize=(13, 5))
    plt.plot(y_daily.index, y_daily.values, linewidth=1.8, label="Real diario (mediana)")
    plt.axvline(train_cut, linestyle="--", linewidth=1)

    plt.plot(pred_daily_median.index, pred_daily_median.values, linewidth=1.2, label=f"{use_col} — mediana diaria (futuro)")
    plt.plot(y_month.index, y_month.values, linewidth=1.5, label="Real mensual (modo aprox)", alpha=0.9)
    plt.plot(pred_monthly_mode.index, pred_monthly_mode.values, linewidth=1.5, label=f"{use_col} — modo mensual (futuro)", alpha=0.9)

    plt.title(f"Codigo {codigo} — Mediana diaria (futuro) + Modo mensual (futuro)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_monthly_max_from_future(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    df_fut: pd.DataFrame,
    out_png: str,
    out_csv: str,
    use_col: str = "pred_ENSEMBLE"
):
    """(4) Mensual MAX de las proyecciones (futuro) junto con real mensual (max o last, aquí uso max real)."""
    if use_col not in df_fut.columns:
        return
    pred = df_fut[use_col].dropna()
    if pred.empty:
        return

    pred_month_max = pred.resample("MS").max()
    y_month_max = y.resample("MS").max()

    pred_month_max.rename("pred_monthly_max").to_frame().reset_index().rename(columns={"index": "mes"}).to_csv(out_csv, index=False)

    plt.figure(figsize=(13, 5))
    plt.plot(y_month_max.index, y_month_max.values, linewidth=2, label="Real mensual (MAX)")
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)
    plt.plot(pred_month_max.index, pred_month_max.values, linewidth=2, label=f"{use_col} — mensual (MAX futuro)")
    plt.title(f"Codigo {codigo} — Mensual MAX de proyecciones (futuro)")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_monthly_bands_intrames(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    df_fut: pd.DataFrame,
    out_png: str,
    out_csv: str,
    use_col: str = "pred_ENSEMBLE",
    q_low: float = 0.10,
    q_high: float = 0.90
):
    """
    (5) Mensual con bandas intrames: mean + [p10, p90] de intradía (futuro),
    junto con real mensual (mean).
    """
    if use_col not in df_fut.columns:
        return
    pred = df_fut[use_col].dropna()
    if pred.empty:
        return

    # real mensual (mean)
    y_month_mean = y.resample("MS").mean()

    # futuro mensual (bandas)
    pred_m_mean = pred.resample("MS").mean()
    pred_m_low  = pred.resample("MS").quantile(q_low)
    pred_m_high = pred.resample("MS").quantile(q_high)

    df_out = pd.DataFrame({
        "mes": pred_m_mean.index,
        "pred_mean": pred_m_mean.values,
        f"pred_q{int(q_low*100)}": pred_m_low.values,
        f"pred_q{int(q_high*100)}": pred_m_high.values
    })
    df_out.to_csv(out_csv, index=False)

    plt.figure(figsize=(13, 5))
    plt.plot(y_month_mean.index, y_month_mean.values, linewidth=2, label="Real mensual (MEDIA)")
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)

    plt.plot(pred_m_mean.index, pred_m_mean.values, linewidth=2, label=f"{use_col} — mensual (MEDIA futuro)")
    plt.fill_between(pred_m_mean.index, pred_m_low.values, pred_m_high.values, alpha=0.2, label=f"Banda {int(q_low*100)}-{int(q_high*100)}% (futuro)")

    plt.title(f"Codigo {codigo} — Mensual con banda intrames (futuro)")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_monthly_close_last(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    df_fut: pd.DataFrame,
    out_png: str,
    out_csv: str = None,
    use_col: str = "pred_ENSEMBLE"
):
    if use_col not in df_fut.columns:
        return
    pred = df_fut[use_col].dropna()
    if pred.empty:
        return

    y_month_last = y.resample("MS").last()
    pred_month_last = pred.resample("MS").last()

    if out_csv is not None:  # ✅ FIX
        pd.DataFrame({
            "mes": pred_month_last.index,
            "pred_month_last": pred_month_last.values
        }).to_csv(out_csv, index=False)

    plt.figure(figsize=(13, 5))
    plt.plot(y_month_last.index, y_month_last.values, linewidth=2, label="Real mensual (CIERRE/LAST)")
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)
    plt.plot(pred_month_last.index, pred_month_last.values, linewidth=2, label=f"{use_col} — cierre mensual (futuro)")
    plt.title(f"Codigo {codigo} — Cierre de mes (LAST) vs proyección")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()



def plot_full_raw(codigo: str, y: pd.Series, out_png: str, max_points_plot=20000):
    plt.figure(figsize=(13,5))
    yp = y
    if len(yp) > max_points_plot:
        step = max(1, len(yp)//max_points_plot)
        yp = yp.iloc[::step]
    plt.plot(yp.index, yp.values, label="Real (raw)", linewidth=1.5)
    plt.title(f"Codigo {codigo} — Serie completa (raw)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_full_plus_forecast_intraday_ensemble(codigo, y, train_cut, df_fut, out_png, col="pred_ENSEMBLE"):
    plt.figure(figsize=(13,5))
    plt.plot(y.index, y.values, label="Real", linewidth=1.6)
    plt.axvline(train_cut, linestyle="--", linewidth=1)

    if col in df_fut.columns:
        plt.plot(df_fut.index, df_fut[col].values, label=f"{col} (forecast intradía)", linewidth=1.2)

    plt.title(f"Codigo {codigo} — Serie completa + forecast intradía ({col})")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def monthly_mode_from_daily(series_daily: pd.Series, decimals=0) -> pd.Series:
    # modo mensual usando redondeo para hacer bins
    sd = series_daily.dropna()
    if sd.empty:
        return sd
    s_round = sd.round(decimals)
    def _mode(x):
        vc = x.value_counts()
        return vc.index[0] if len(vc) else np.nan
    return s_round.resample("MS").apply(_mode)

def plot_daily_median_and_monthly_mode(codigo, y, df_fut, train_cut, out_png, col="pred_ENSEMBLE", mode_decimals=0):
    # Real diario (mediana intradía del día)
    y_day = y.resample("D").median()

    # Futuro: mediana diaria de predicción
    if col not in df_fut.columns:
        return
    p_day_median = df_fut[col].resample("D").median()

    # Modo mensual de esa mediana diaria
    p_month_mode = monthly_mode_from_daily(p_day_median, decimals=mode_decimals)

    plt.figure(figsize=(13,5))
    plt.plot(y_day.index, y_day.values, label="Real (mediana diaria)", linewidth=1.7)
    plt.axvline(train_cut, linestyle="--", linewidth=1)

    plt.plot(p_day_median.index, p_day_median.values, label=f"{col} (mediana diaria futuro)", linewidth=1.2)
    plt.plot(p_month_mode.index, p_month_mode.values, label=f"{col} (MODO mensual de mediana diaria)", linewidth=2.0)

    plt.title(f"Codigo {codigo} — Mediana diaria + modo mensual (desde intradía)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =========================================================
# EXPORTS WIDE MENSUAL (MEDIA / MEDIANA)
# =========================================================
def month_labels(idx: pd.DatetimeIndex) -> List[str]:
    # formato tipo "Dec-25"
    return [pd.Timestamp(x).strftime("%b-%y") for x in pd.DatetimeIndex(idx)]

def monthly_wide_per_model_from_future(
    codigo: str,
    df_fut_intraday: pd.DataFrame,   # index=future_idx, columnas pred_<MODEL>
    agg: str = "mean"                # "mean" o "median"
) -> pd.DataFrame:
    """
    Devuelve DataFrame wide:
      columnas: codigo | Dec-25 | Jan-26 | ... (mensual)
      FILAS: una por modelo (columna pred_<MODEL>)
    """
    pred_cols = [c for c in df_fut_intraday.columns if c.startswith("pred_") and c != "pred_ENSEMBLE"]
    if not pred_cols:
        return pd.DataFrame()

    rows = []
    for c in pred_cols:
        ser = df_fut_intraday[c].dropna()
        if ser.empty:
            continue

        if agg == "mean":
            mser = ser.resample("MS").mean()
        elif agg == "median":
            mser = ser.resample("MS").median()
        else:
            raise ValueError("agg debe ser 'mean' o 'median'")

        # wide dict: codigo + meses
        row = {"codigo": str(codigo), "modelo": c.replace("pred_", "")}
        for lab, val in zip(month_labels(mser.index), mser.values):
            row[lab] = float(val) if np.isfinite(val) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    dfw = pd.DataFrame(rows)

    # ordena columnas: codigo, modelo, meses...
    fixed = ["codigo", "modelo"]
    month_cols = [c for c in dfw.columns if c not in fixed]
    # ordenar meses por tiempo usando un truco (parse "Mon-YY")
    def _key(mon):
        return pd.to_datetime("01-" + mon, format="%d-%b-%y", errors="coerce")
    month_cols = sorted(month_cols, key=_key)
    return dfw[fixed + month_cols]


def monthly_wide_best_model_from_future(
    codigo: str,
    best_model: str,                 # por ejemplo "MLP", "TCN", "ETS", etc.
    df_fut_intraday: pd.DataFrame,
    agg: str = "mean"                # "mean" o "median"
) -> pd.DataFrame:
    """
    Devuelve 1 fila wide:
      codigo | modelo | Dec-25 | Jan-26 | ...
    usando SOLO el mejor modelo.
    """
    col = f"pred_{best_model}"
    if col not in df_fut_intraday.columns:
        return pd.DataFrame()

    ser = df_fut_intraday[col].dropna()
    if ser.empty:
        return pd.DataFrame()

    if agg == "mean":
        mser = ser.resample("MS").mean()
    elif agg == "median":
        mser = ser.resample("MS").median()
    else:
        raise ValueError("agg debe ser 'mean' o 'median'")

    row = {"codigo": str(codigo), "modelo": str(best_model)}
    for lab, val in zip(month_labels(mser.index), mser.values):
        row[lab] = float(val) if np.isfinite(val) else np.nan

    dfw = pd.DataFrame([row])

    fixed = ["codigo", "modelo"]
    month_cols = [c for c in dfw.columns if c not in fixed]
    def _key(mon):
        return pd.to_datetime("01-" + mon, format="%d-%b-%y", errors="coerce")
    month_cols = sorted(month_cols, key=_key)
    return dfw[fixed + month_cols]


# =========================================================
# FILTRO FUERTE DE OUTLIERS (reemplaza tu filter_models_by_scale)
# =========================================================
def robust_abs_scale(x: np.ndarray, q: float = 0.95) -> float:
    x = np.asarray(x, float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.nanquantile(np.abs(x), q))

def drop_outlier_models(
    y_ref: pd.Series,
    preds: Dict[str, Optional[np.ndarray]],
    ratio: float = 30.0,          # más estricto que 50
    q_ref: float = 0.95,
    q_pred: float = 0.995,        # detecta "picos"
    max_nan_frac: float = 0.05
) -> Dict[str, Optional[np.ndarray]]:
    """
    Elimina modelos que:
      - tienen demasiados NaNs
      - o tienen escala (cuantil abs) demasiado grande vs la serie real
      - o tienen picos extremos (q=0.995) demasiado grandes
    """
    y_scale = robust_abs_scale(y_ref.values, q=q_ref)
    if not np.isfinite(y_scale) or y_scale <= 0:
        return preds

    out = dict(preds)
    for name, p in preds.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)
        if p.size == 0:
            out[name] = None
            continue

        nan_frac = float(np.mean(~np.isfinite(p)))
        if nan_frac > max_nan_frac:
            out[name] = None
            continue

        p_f = p[np.isfinite(p)]
        if p_f.size == 0:
            out[name] = None
            continue

        # escala típica
        p_scale = robust_abs_scale(p_f, q=q_ref)
        # picos (si hay 1-2 picos, esto lo detecta)
        p_peak  = robust_abs_scale(p_f, q=q_pred)

        if (np.isfinite(p_scale) and p_scale > ratio * y_scale) or (np.isfinite(p_peak) and p_peak > ratio * y_scale):
            out[name] = None

    return out


# =========================================================
# PLOTS NUEVOS (4 gráficos)
# =========================================================
def plot_forecast_only_intraday(
    codigo: str,
    y: pd.Series,
    train_cut: pd.Timestamp,
    future_idx: pd.DatetimeIndex,
    preds_future: Dict[str, Optional[np.ndarray]],
    out_png: str,
    max_points_plot: int = 12000
):
    """
    (A) Serie completa + forecast intradía al final (tipo plot_forecast_only_101101)
    """
    plt.figure(figsize=(13, 5))

    # downsample real completo
    y_plot = y
    if len(y_plot) > max_points_plot:
        step = max(1, len(y_plot)//max_points_plot)
        y_plot = y_plot.iloc[::step]
    plt.plot(y_plot.index, y_plot.values, label="Real", linewidth=1.8)

    plt.axvline(train_cut, linestyle="--", linewidth=1)

    # filtrar modelos outliers usando referencia REAL COMPLETA
    pf = drop_outlier_models(
        y_ref=y,
        preds=preds_future,
        ratio=30.0,
        q_ref=0.95,
        q_pred=0.995
    )

    for name, p in pf.items():
        if p is None:
            continue
        p = np.asarray(p, float).reshape(-1)

        # downsample forecast (solo futuro)
        fidx = future_idx
        if len(fidx) > max_points_plot:
            step = max(1, len(fidx)//max_points_plot)
            fidx = fidx[::step]
            p = p[::step]

        m = min(len(fidx), len(p))
        if m <= 0:
            continue

        plt.plot(fidx[:m], p[:m], label=f"{name} (forecast)", linewidth=1)

    plt.title(f"Codigo {codigo} — Serie completa + forecast intradía (pasos={len(future_idx)})")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_test_vs_pred_intraday_strict(
    codigo: str,
    y_test: pd.Series,
    preds_test_series: Dict[str, Optional[pd.Series]],
    train_cut: pd.Timestamp,
    out_png: str,
    max_points_plot: int = 12000
):
    """
    (B) TEST vs PRED intradía, pero con filtro fuerte de outliers en preds.
    """
    plt.figure(figsize=(13, 5))

    # downsample real test
    y_plot = y_test
    if len(y_plot) > max_points_plot:
        step = max(1, len(y_plot)//max_points_plot)
        y_plot = y_plot.iloc[::step]
    plt.plot(y_plot.index, y_plot.values, label="Real (TEST)", linewidth=2)

    plt.axvline(train_cut, linestyle="--", linewidth=1)

    # convertir series a arrays para filtrar outliers
    preds_arr = {}
    for name, s in preds_test_series.items():
        preds_arr[name] = None if s is None else np.asarray(s.values, float)

    pf = drop_outlier_models(
        y_ref=y_test,
        preds=preds_arr,
        ratio=30.0,
        q_ref=0.95,
        q_pred=0.995
    )

    for name, s in preds_test_series.items():
        if s is None:
            continue
        if pf.get(name, None) is None:
            continue

        # alinear por índice con y_test
        common_idx = y_test.index.intersection(s.index)
        if len(common_idx) == 0:
            continue
        sp = s.loc[common_idx]

        # downsample pred
        if len(sp) > max_points_plot:
            step = max(1, len(sp)//max_points_plot)
            sp = sp.iloc[::step]

        plt.plot(sp.index, sp.values, label=f"{name} (pred test)", linewidth=1)

    plt.title(f"Codigo {codigo} — TEST vs PRED (intradía)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_monthly_from_intraday_preds(
    codigo: str,
    y: pd.Series,
    df_fut_intraday: pd.DataFrame,
    train_cut: pd.Timestamp,
    out_png_mean: str,
    out_png_median: str
):
    # Asegurar DatetimeIndex
    y = y.copy()
    y.index = pd.DatetimeIndex(y.index)

    df = df_fut_intraday.copy()
    df.index = pd.DatetimeIndex(df.index)

    # Real mensual
    y_month = y.resample("MS").mean()

    # columnas pred_
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No hay columnas pred_ en df_fut_intraday")

    # limpiar predicciones (por si acaso)
    for c in pred_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].interpolate(limit_direction="both").ffill().bfill()

    # filtrar outliers (si quieres)
    preds_future = {c.replace("pred_", ""): df[c].values for c in pred_cols}
    pf = drop_outlier_models(y_ref=y, preds=preds_future, ratio=30.0, q_ref=0.95, q_pred=0.995)
    keep_cols = [f"pred_{m}" for m, v in pf.items() if v is not None and f"pred_{m}" in df.columns]

    if not keep_cols:
        raise ValueError("Después del filtro outlier no quedó ningún modelo para graficar.")

    df_keep = df[keep_cols].copy()

    # índice mensual futuro (seguro)
    start_m = pd.Timestamp(df_keep.index.min()).to_period("M").start_time
    end_m   = pd.Timestamp(df_keep.index.max()).to_period("M").start_time
    if pd.isna(start_m) or pd.isna(end_m):
        raise ValueError("start_m/end_m salió NaT")

    month_idx = pd.date_range(start=start_m, end=end_m, freq="MS")
    if len(month_idx) == 0:
        raise ValueError("month_idx quedó vacío")

    # ----- MEDIA -----
    df_month_mean = pd.DataFrame(index=month_idx)
    for c in keep_cols:
        s = df_keep[c].dropna()
        df_month_mean[c] = s.resample("MS").mean().reindex(month_idx)

    # ensemble mean
    df_month_mean["pred_ENSEMBLE"] = df_month_mean[keep_cols].mean(axis=1)

    plt.figure(figsize=(13, 5))
    plt.plot(y_month.index, y_month.values, label="Real mensual (promedio)", linewidth=2)
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)

    for c in df_month_mean.columns:
        if df_month_mean[c].notna().sum() == 0:
            continue
        plt.plot(df_month_mean.index, df_month_mean[c].values, label=c)

    plt.title(f"Codigo {codigo} — Mensual (MEDIA) desde intradía")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png_mean, dpi=150)
    plt.close()

    # ----- MEDIANA -----
    df_month_median = pd.DataFrame(index=month_idx)
    for c in keep_cols:
        s = df_keep[c].dropna()
        df_month_median[c] = s.resample("MS").median().reindex(month_idx)

    # ensemble median
    df_month_median["pred_ENSEMBLE"] = df_month_median[keep_cols].median(axis=1)

    plt.figure(figsize=(13, 5))
    plt.plot(y_month.index, y_month.values, label="Real mensual (promedio)", linewidth=2)
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)

    for c in df_month_median.columns:
        if df_month_median[c].notna().sum() == 0:
            continue
        plt.plot(df_month_median.index, df_month_median[c].values, label=c)

    plt.title(f"Codigo {codigo} — Mensual (MEDIANA) desde intradía")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png_median, dpi=150)
    plt.close()

def plot_monthly_max_future(codigo, y, df_fut, train_cut, out_png, col="pred_ENSEMBLE"):
    y_m_max = y.resample("MS").max()
    if col not in df_fut.columns:
        return
    p_m_max = df_fut[col].resample("MS").max()

    plt.figure(figsize=(13,5))
    plt.plot(y_m_max.index, y_m_max.values, label="Real mensual (MAX)", linewidth=2)
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)
    plt.plot(p_m_max.index, p_m_max.values, label=f"{col} — mensual (MAX futuro)", linewidth=2)

    plt.title(f"Codigo {codigo} — Mensual MAX (real vs futuro)")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_monthly_with_intramonth_bands(codigo, y, df_fut, train_cut, out_png, col="pred_ENSEMBLE", qlo=0.10, qhi=0.90):
    y_m_mean = y.resample("MS").mean()
    if col not in df_fut.columns:
        return

    p_m_mean = df_fut[col].resample("MS").mean()
    p_m_qlo  = df_fut[col].resample("MS").quantile(qlo)
    p_m_qhi  = df_fut[col].resample("MS").quantile(qhi)

    plt.figure(figsize=(13,5))
    plt.plot(y_m_mean.index, y_m_mean.values, label="Real mensual (MEDIA)", linewidth=2)
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)

    plt.plot(p_m_mean.index, p_m_mean.values, label=f"{col} — mensual (MEDIA futuro)", linewidth=2)
    plt.fill_between(p_m_mean.index, p_m_qlo.values, p_m_qhi.values, alpha=0.2, label=f"Banda {int(qlo*100)}-{int(qhi*100)}% (futuro)")

    plt.title(f"Codigo {codigo} — Mensual con banda intrames (futuro)")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_monthly_close_last_SIMPLE(codigo, y, df_fut, train_cut, out_png, col="pred_ENSEMBLE"):

    y_last = y.resample("MS").last()
    if col not in df_fut.columns:
        return
    p_last = df_fut[col].resample("MS").last()

    plt.figure(figsize=(13,5))
    plt.plot(y_last.index, y_last.values, label="Real mensual (CIERRE/LAST)", linewidth=2)
    plt.axvline(train_cut.to_period("M").start_time, linestyle="--", linewidth=1)
    plt.plot(p_last.index, p_last.values, label=f"{col} — cierre mensual (futuro)", linewidth=2)

    plt.title(f"Codigo {codigo} — Cierre de mes (LAST) vs proyección")
    plt.xlabel("Mes")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =========================================================
# Read LONG + serie intradía fija
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

def intraday_series(df_long: pd.DataFrame, codigo: str, base_freq: str) -> pd.Series:
    g = df_long[df_long["codigo"] == codigo].copy()
    if g.empty:
        return pd.Series(dtype=float)
    s = pd.Series(g["valor"].values, index=pd.to_datetime(g["fecha"])).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.asfreq(base_freq)
    s = s.interpolate(limit_direction="both")
    s = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return s


# =========================================================
# CORE
# =========================================================
def run_for_codigo_intraday(codigo: str, y: pd.Series, base_freq: str):
    """
    Corre modelos intradía para un código, guarda:
    - plot test vs pred (intradía)
    - plot serie completa + pred futuro intradía
    - plots mensuales (mean/median) desde intradía
    - exports wide mensuales
    - NUEVAS gráficas (1..6) usando pred_ENSEMBLE robusto
    """

    # =========================================================
    # Helpers (si ya las tienes en otro lado, puedes borrar estas)
    # =========================================================
    def _safe_quantile(arr: np.ndarray, q: float):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.quantile(arr, q))

    def drop_outlier_models(
        y_ref: pd.Series,
        preds: Dict[str, np.ndarray],
        ratio: float = 30.0,
        q_ref: float = 0.95,
        q_pred: float = 0.995,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Filtro simple: elimina modelos cuya escala se dispara vs histórico.
        Compara cuantiles (no máximos) para evitar picos.
        """
        yv = np.asarray(y_ref.values, float)
        ref_q = _safe_quantile(yv, q_ref)
        if not np.isfinite(ref_q) or ref_q == 0:
            # fallback: usa mediana
            ref_q = _safe_quantile(yv, 0.50)

        kept: Dict[str, Optional[np.ndarray]] = {}
        for name, arr in preds.items():
            if arr is None:
                kept[name] = None
                continue
            a = np.asarray(arr, float).reshape(-1)
            qv = _safe_quantile(a, q_pred)
            if not np.isfinite(qv):
                kept[name] = None
                continue

            # Si el ref_q es muy pequeño, solo revisa que no sea absurdamente grande
            if abs(ref_q) < 1e-9:
                kept[name] = None if abs(qv) > 1e6 else a
                continue

            if abs(qv) > ratio * abs(ref_q):
                kept[name] = None
            else:
                kept[name] = a
        return kept

    # =========================================================
    # (0) Preparación
    # =========================================================
    y = y.dropna().astype(float)
    if len(y) < (max(LOOKBACK_NN, LAGS_TABULAR) + 500):
        print(f"[SKIP] {codigo}: serie muy corta (n={len(y)})")
        return None

    all_zero = is_all_zero_series(y)

    train_end_i = int(len(y) * TRAIN_RATIO)
    min_train = max(LOOKBACK_NN, LAGS_TABULAR) + 1000
    train_end_i = max(train_end_i, min_train)
    train_end_i = min(train_end_i, len(y) - MIN_TEST_POINTS)

    if train_end_i <= min_train or train_end_i >= len(y) - MIN_TEST_POINTS:
        print(f"[SKIP] {codigo}: split inválido (n={len(y)}, train_end_i={train_end_i})")
        return None

    train_cut = y.index[train_end_i]
    y_train = y.iloc[:train_end_i]
    y_test  = y.iloc[train_end_i:]

    if len(y_train) < min_train or len(y_test) < MIN_TEST_POINTS:
        print(f"[SKIP] {codigo}: train/test insuficiente (train={len(y_train)}, test={len(y_test)})")
        return None

    test_idx = y_test.index
    test_len = len(y_test)

    steps_per_day = steps_per_day_from_freq(base_freq)
    H_FUTURE_STEPS = int(FUTURE_DAYS * steps_per_day)

    future_idx = pd.date_range(
        start=y.index[-1] + pd.tseries.frequencies.to_offset(base_freq),
        periods=H_FUTURE_STEPS,
        freq=base_freq
    )

    seasonal_periods = steps_per_day

    preds_test: Dict[str, Optional[pd.Series]] = {}
    preds_fut:  Dict[str, Optional[np.ndarray]] = {}
    scores: Dict[str, float] = {}

    model_names_all = ["ETS","SARIMAX","TCN","LSTM","DL_MultiTask","Linear","Ridge","Lasso","MLP","HGB","LGBM"]

    # =========================================================
    # (1) Modelado
    # =========================================================
    if all_zero:
        for name in model_names_all:
            preds_test[name] = pd.Series(np.zeros(test_len, float), index=test_idx)
            preds_fut[name]  = np.zeros(len(future_idx), float)
            scores[name]     = 0.0

    else:
        # ETS
        if RUN_ETS:
            try:
                yhat_test, yhat_fut = fit_predict_ets(
                    codigo, y_train, test_len, len(future_idx),
                    seasonal_periods, base_freq, train_cut
                )
                preds_test["ETS"] = as_series(yhat_test, test_idx)
                preds_fut["ETS"]  = yhat_fut
                s = preds_test["ETS"]
                scores["ETS"] = rmse(y_test.values, s.values) if s is not None else float("nan")
            except Exception as e:
                print(f"[WARN] {codigo} ETS falló: {e}")
                preds_test["ETS"] = None
                preds_fut["ETS"]  = None

        # SARIMAX
        if RUN_SARIMAX:
            try:
                yhat_test, yhat_fut = fit_predict_sarimax(
                    codigo, y_train, test_len, len(future_idx),
                    seasonal_periods, base_freq, train_cut
                )
                preds_test["SARIMAX"] = as_series(yhat_test, test_idx)
                preds_fut["SARIMAX"]  = yhat_fut
                s = preds_test["SARIMAX"]
                scores["SARIMAX"] = rmse(y_test.values, s.values) if s is not None else float("nan")
            except Exception as e:
                print(f"[WARN] {codigo} SARIMAX falló: {e}")
                preds_test["SARIMAX"] = None
                preds_fut["SARIMAX"]  = None

        # Tabulares
        def try_tab(model_kind, key=None):
            k = key or model_kind
            try:
                yhat_test, yhat_fut = fit_predict_tabular_model(
                    codigo=codigo,
                    y=y,
                    train_cut=train_cut,
                    test_len=test_len,
                    future_idx=future_idx,
                    lags=LAGS_TABULAR,
                    model_kind=model_kind,
                    base_freq=base_freq,
                )
                preds_test[k] = as_series(yhat_test, test_idx)
                preds_fut[k]  = yhat_fut
                s_k = preds_test.get(k)
                scores[k] = rmse(y_test.values, s_k.values) if s_k is not None else float("nan")
            except Exception as e:
                print(f"[WARN] {codigo} {k} falló: {e}")
                preds_test[k] = None
                preds_fut[k]  = None
            finally:
                cleanup_memory(f"{codigo}: after TAB_{k}")

        if RUN_LINEAR: try_tab("Linear")
        if RUN_RIDGE:  try_tab("Ridge")
        if RUN_LASSO:  try_tab("Lasso")
        if RUN_MLP:    try_tab("MLP")
        if RUN_HGB:    try_tab("HGB")
        if RUN_LGBM:   try_tab("LGBM", key="LGBM")

        # NNs (asinh scaler + supervised)
        sc = fit_asinh_scaler(y_train.values)
        y_scaled_all = transform_asinh(y.values, sc)

        X_all, Y_all = make_supervised(y_scaled_all, LOOKBACK_NN)
        target_idx = y.index[LOOKBACK_NN:]

        train_mask = target_idx < train_cut
        test_mask  = target_idx >= train_cut

        Xtr, Ytr = X_all[train_mask], Y_all[train_mask]
        Xte_all = X_all[test_mask]
        test_idx_nn_all = target_idx[test_mask]

        # alineamos a test_len
        Xte = Xte_all[:test_len]
        test_idx_nn = test_idx_nn_all[:test_len]

        ntr = len(Xtr)
        nval = max(200, int(0.1 * ntr))
        Xtrain, Ytrain = Xtr[:-nval], Ytr[:-nval]
        Xval,   Yval   = Xtr[-nval:], Ytr[-nval:]

        last_window = y_scaled_all[-LOOKBACK_NN:]

        # TCN
        if RUN_TCN:
            try:
                params = {
                    "lookback": LOOKBACK_NN,
                    "epochs": NN_EPOCHS,
                    "batch": NN_BATCH,
                    "patience": NN_PATIENCE,
                    "train_cut": str(train_cut),
                    "freq": base_freq,
                    "arch": "TCN64_k3_dil[1,2,4,8,16,32]",
                }
                m, _ = load_or_train_keras(
                    codigo=codigo,
                    model_name="TCN",
                    build_fn=build_tcn,
                    train_fn=train_keras_model,
                    Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                    params=params
                )
                yhat_test_s = predict_keras_one_step(m, Xte)
                yhat_test = inverse_asinh(yhat_test_s, sc)

                yhat_fut_s = forecast_keras_autoregressive(m, last_window, len(future_idx))
                yhat_fut = inverse_asinh(yhat_fut_s, sc)

                preds_test["TCN"] = as_series(yhat_test, test_idx_nn)
                preds_fut["TCN"]  = yhat_fut
                scores["TCN"] = rmse(y.loc[preds_test["TCN"].index].values, preds_test["TCN"].values)
            except Exception as e:
                print(f"[WARN] {codigo} TCN falló: {e}")
                preds_test["TCN"] = None
                preds_fut["TCN"]  = None
            finally:
                cleanup_memory(f"{codigo}: after TCN")

        # LSTM
        if RUN_LSTM:
            try:
                params = {
                    "lookback": LOOKBACK_NN,
                    "epochs": NN_EPOCHS,
                    "batch": NN_BATCH,
                    "patience": NN_PATIENCE,
                    "train_cut": str(train_cut),
                    "freq": base_freq,
                    "arch": "LSTM(64,32)+Dense16",
                }
                m, _ = load_or_train_keras(
                    codigo=codigo,
                    model_name="LSTM",
                    build_fn=build_lstm,
                    train_fn=train_keras_model,
                    Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                    params=params
                )
                yhat_test_s = predict_keras_one_step(m, Xte)
                yhat_test = inverse_asinh(yhat_test_s, sc)

                yhat_fut_s = forecast_keras_autoregressive(m, last_window, len(future_idx))
                yhat_fut = inverse_asinh(yhat_fut_s, sc)

                preds_test["LSTM"] = as_series(yhat_test, test_idx_nn)
                preds_fut["LSTM"]  = yhat_fut
                scores["LSTM"] = rmse(y.loc[preds_test["LSTM"].index].values, preds_test["LSTM"].values)
            except Exception as e:
                print(f"[WARN] {codigo} LSTM falló: {e}")
                preds_test["LSTM"] = None
                preds_fut["LSTM"]  = None
            finally:
                cleanup_memory(f"{codigo}: after LSTM")

        # MultiTask
        if RUN_MULTITASK_DL:
            try:
                params = {
                    "lookback": LOOKBACK_NN,
                    "epochs": NN_EPOCHS,
                    "batch": NN_BATCH,
                    "patience": NN_PATIENCE,
                    "train_cut": str(train_cut),
                    "freq": base_freq,
                    "arch": "Conv1D+Conv1D+GAP (2 heads)",
                }
                m, _ = load_or_train_keras(
                    codigo=codigo,
                    model_name="DL_MultiTask",
                    build_fn=build_multitask_dl,
                    train_fn=train_keras_model,
                    Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                    params=params
                )
                yhat_test_s = predict_keras_one_step(m, Xte)
                yhat_test = inverse_asinh(yhat_test_s, sc)

                yhat_fut_s = forecast_keras_autoregressive(m, last_window, len(future_idx))
                yhat_fut = inverse_asinh(yhat_fut_s, sc)

                preds_test["DL_MultiTask"] = as_series(yhat_test, test_idx_nn)
                preds_fut["DL_MultiTask"]  = yhat_fut
                scores["DL_MultiTask"] = rmse(
                    y.loc[preds_test["DL_MultiTask"].index].values,
                    preds_test["DL_MultiTask"].values
                )
            except Exception as e:
                print(f"[WARN] {codigo} MultiTask DL falló: {e}")
                preds_test["DL_MultiTask"] = None
                preds_fut["DL_MultiTask"]  = None
            finally:
                cleanup_memory(f"{codigo}: after DL_MultiTask")

    scored = sorted(scores.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e99)
    print(f"\n=== {codigo} INTRADIA n={len(y)} test={len(y_test)} steps_future={len(future_idx)} freq={base_freq} ===")
    for name, r in scored:
        print(f"  {name:14s} RMSE_test = {r:,.4f}")

    # =========================================================
    # (2) Salidas por código
    # =========================================================
    out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
    ensure_dir(out_c_dir)

    # (B) TEST vs PRED intradía
    out_png_test = os.path.join(out_c_dir, f"plot_test_vs_pred_intraday_{codigo}.png")
    plot_test_vs_pred_intraday_strict(
        codigo=codigo,
        y_test=y_test,
        preds_test_series=preds_test,
        train_cut=train_cut,
        out_png=out_png_test
    )
    print(f"[OK] guardado TEST plot: {out_png_test}")

    # =========================================================
    # (3) Construir df_fut y ENSEMBLE ROBUSTO (CLAVE)
    # =========================================================
    df_fut = pd.DataFrame(index=future_idx)
    for name, arr in preds_fut.items():
        if arr is None:
            continue
        arr = np.asarray(arr, float).reshape(-1)
        df_fut[f"pred_{name}"] = pd.Series(arr[:len(future_idx)], index=future_idx)

    pred_model_cols = [c for c in df_fut.columns if c.startswith("pred_") and c != "pred_ENSEMBLE"]

    # limpieza base
    for c in pred_model_cols:
        df_fut[c] = pd.to_numeric(df_fut[c], errors="coerce")
        df_fut[c] = df_fut[c].replace([np.inf, -np.inf], np.nan)
        if df_fut[c].notna().sum() > 10:
            df_fut[c] = df_fut[c].interpolate(limit_direction="both").ffill().bfill()

    # clip a rango histórico (evita 1e296 finito)
    hist_q01 = y.quantile(0.01)
    hist_q99 = y.quantile(0.99)
    iqr = hist_q99 - hist_q01
    lo = hist_q01 - 3 * iqr
    hi = hist_q99 + 3 * iqr
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        for c in pred_model_cols:
            df_fut[c] = df_fut[c].clip(lo, hi)

    # filtrado por escala + ensemble robusto
    if pred_model_cols:
        preds_dict = {c.replace("pred_", ""): df_fut[c].values for c in pred_model_cols}
        keep = drop_outlier_models(y_ref=y, preds=preds_dict, ratio=30.0, q_ref=0.95, q_pred=0.995)
        keep_cols = [f"pred_{m}" for m, v in keep.items() if v is not None and f"pred_{m}" in df_fut.columns]

        if keep_cols:
            df_fut["pred_ENSEMBLE_MEDIAN"] = df_fut[keep_cols].median(axis=1)
            df_fut["pred_ENSEMBLE_MEAN"]   = df_fut[keep_cols].mean(axis=1)
            df_fut["pred_ENSEMBLE"]        = df_fut["pred_ENSEMBLE_MEDIAN"]  # ✅ USAR ESTE
        else:
            df_fut["pred_ENSEMBLE"] = df_fut[pred_model_cols].median(axis=1)
    else:
        # si no hay modelos, deja vacío (pero no debería pasar)
        pass

    # logs de sanidad
    if pred_model_cols:
        print("NaN ratio por modelo (futuro):")
        print(df_fut[pred_model_cols].isna().mean().sort_values(ascending=False).head(20))
        print("\nConteo de finitos por modelo:")
        print(np.isfinite(df_fut[pred_model_cols].values).sum(axis=0))

    # guardar CSV futuro intradía
    out_csv_future = os.path.join(out_c_dir, f"future_intraday_{codigo}.csv")
    df_fut.reset_index().rename(columns={"index": "fecha"}).to_csv(out_csv_future, index=False)
    print(f"[OK] guardado: {out_csv_future}")

    # =========================================================
    # (A) Serie completa + forecast intradía (solo futuro)
    # =========================================================
    out_png_forecast_only = os.path.join(out_c_dir, f"plot_forecast_only_{codigo}.png")
    preds_future_for_plot = {
        c.replace("pred_", ""): df_fut[c].values
        for c in pred_model_cols
        if c in df_fut.columns
    }
    plot_forecast_only_intraday(
        codigo=codigo,
        y=y,
        train_cut=train_cut,
        future_idx=future_idx,
        preds_future=preds_future_for_plot,
        out_png=out_png_forecast_only
    )
    print(f"[OK] guardado forecast-only: {out_png_forecast_only}")

    # =========================================================
    # (C) y (D) Mensual: MEDIA y MEDIANA desde intradía (usando df_fut ya saneado)
    # =========================================================
    out_png_month_mean   = os.path.join(out_c_dir, f"plot_monthly_MEAN_{codigo}.png")
    out_png_month_median = os.path.join(out_c_dir, f"plot_monthly_MEDIAN_{codigo}.png")
    try:
        plot_monthly_from_intraday_preds(
            codigo=codigo,
            y=y,
            df_fut_intraday=df_fut,
            train_cut=train_cut,
            out_png_mean=out_png_month_mean,
            out_png_median=out_png_month_median
        )
        print(f"[OK] guardado mensual MEAN: {out_png_month_mean}")
        print(f"[OK] guardado mensual MEDIAN: {out_png_month_median}")
    except Exception as e:
        print(f"[WARN] {codigo}: no pude generar plots mensuales: {e}")

    # =========================================================
    # (4) Elegir mejor modelo por RMSE (con fallback a ENSEMBLE robusto)
    # =========================================================
    candidates = []
    for name, r in scores.items():
        if not np.isfinite(r):
            continue
        if f"pred_{name}" not in df_fut.columns:
            continue
        candidates.append((name, r))

    best_model = None
    if candidates:
        candidates.sort(key=lambda x: x[1])
        best_model = candidates[0][0]
    else:
        if "pred_ENSEMBLE" in df_fut.columns:
            best_model = "ENSEMBLE"

    if best_model is not None:
        with open(os.path.join(out_c_dir, f"best_model_{codigo}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{best_model}\n")

    # =========================================================
    # (5) EXPORT WIDE mensual por modelo (mean/median)
    # =========================================================
    df_wide_mean = monthly_wide_per_model_from_future(codigo, df_fut, agg="mean")
    df_wide_median = monthly_wide_per_model_from_future(codigo, df_fut, agg="median")

    out_wide_mean = os.path.join(out_c_dir, f"future_monthly_MEAN_wide_{codigo}.csv")
    out_wide_median = os.path.join(out_c_dir, f"future_monthly_MEDIAN_wide_{codigo}.csv")

    if not df_wide_mean.empty:
        df_wide_mean.to_csv(out_wide_mean, index=False, encoding="utf-8")
        print(f"[OK] guardado WIDE mensual MEAN por modelo: {out_wide_mean}")
    else:
        print(f"[WARN] {codigo}: no pude crear WIDE mensual MEAN (sin predicciones)")

    if not df_wide_median.empty:
        df_wide_median.to_csv(out_wide_median, index=False, encoding="utf-8")
        print(f"[OK] guardado WIDE mensual MEDIAN por modelo: {out_wide_median}")
    else:
        print(f"[WARN] {codigo}: no pude crear WIDE mensual MEDIAN (sin predicciones)")

    # =========================================================
    # (6) NUEVAS GRAFICAS (1..6) — usa pred_ENSEMBLE ROBUSTO
    # =========================================================
    out_png_raw = os.path.join(out_c_dir, f"plot_full_raw_{codigo}.png")
    plot_full_series_raw(codigo=codigo, y=y, out_png=out_png_raw)
    print(f"[OK] guardado serie raw: {out_png_raw}")

    out_png_full_plus_future = os.path.join(out_c_dir, f"plot_full_plus_future_intraday_{codigo}.png")
    plot_full_plus_future_intraday_only(
        codigo=codigo,
        y=y,
        train_cut=train_cut,
        df_fut=df_fut,
        out_png=out_png_full_plus_future,
        plot_models=False,       # recomiendo False para no saturar
        plot_ensemble=True
    )
    print(f"[OK] guardado serie + futuro intradía: {out_png_full_plus_future}")

    # (3) mediana diaria + modo mensual
    out_png_daily_median_month_mode = os.path.join(out_c_dir, f"plot_daily_median_and_monthly_mode_{codigo}.png")
    out_csv_daily_median = os.path.join(out_c_dir, f"future_daily_median_{codigo}.csv")
    out_csv_month_mode = os.path.join(out_c_dir, f"future_monthly_mode_{codigo}.csv")
    plot_daily_median_and_monthly_mode_from_future(
        codigo=codigo,
        y=y,
        train_cut=train_cut,
        df_fut=df_fut,
        out_png=out_png_daily_median_month_mode,
        out_csv_daily=out_csv_daily_median,
        out_csv_monthly=out_csv_month_mode,
        use_col="pred_ENSEMBLE",
        mode_round=2
    )
    print(f"[OK] guardado daily median + monthly mode: {out_png_daily_median_month_mode}")

    # (4) mensual MAX futuro
    out_png_month_max = os.path.join(out_c_dir, f"plot_monthly_MAX_future_{codigo}.png")
    out_csv_month_max = os.path.join(out_c_dir, f"future_monthly_MAX_{codigo}.csv")
    plot_monthly_max_from_future(
        codigo=codigo,
        y=y,
        train_cut=train_cut,
        df_fut=df_fut,
        out_png=out_png_month_max,
        out_csv=out_csv_month_max,
        use_col="pred_ENSEMBLE"
    )
    print(f"[OK] guardado mensual MAX futuro: {out_png_month_max}")

    # (5) bandas intrames (p10-p90) + mean
    out_png_bands = os.path.join(out_c_dir, f"plot_monthly_BANDS_{codigo}.png")
    out_csv_bands = os.path.join(out_c_dir, f"future_monthly_BANDS_{codigo}.csv")
    plot_monthly_bands_intrames(
        codigo=codigo,
        y=y,
        train_cut=train_cut,
        df_fut=df_fut,
        out_png=out_png_bands,
        out_csv=out_csv_bands,
        use_col="pred_ENSEMBLE",
        q_low=0.10,
        q_high=0.90
    )
    print(f"[OK] guardado bandas intrames: {out_png_bands}")

    # (6) cierre de mes (LAST)
    out_png_close = os.path.join(out_c_dir, f"plot_monthly_CLOSE_LAST_{codigo}.png")
    out_csv_close = os.path.join(out_c_dir, f"future_monthly_CLOSE_LAST_{codigo}.csv")
    plot_monthly_close_last(
        codigo=codigo,
        y=y,
        train_cut=train_cut,
        df_fut=df_fut,
        out_png=out_png_close,
        out_csv=out_csv_close,
        use_col="pred_ENSEMBLE"
    )
    print(f"[OK] guardado cierre mensual (LAST): {out_png_close}")

    # =========================================================
    # Devolver
    # =========================================================
    return {
        "codigo": str(codigo),
        "best_model": best_model,
        "df_fut": df_fut,
        "scores": scores
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

    # acumuladores globales
    all_best_mean = []
    all_best_median = []

    for c in codigos:
        try:
            g = df_long[df_long["codigo"] == str(c)].copy()
            if g.empty:
                continue
            idx = pd.to_datetime(g["fecha"]).sort_values()
            base_freq = BASE_FREQ or infer_base_freq(pd.DatetimeIndex(idx))
            y = intraday_series(df_long, str(c), base_freq=base_freq)
            if y.empty:
                print(f"[SKIP] {c}: serie intradía vacía")
                continue

            result = run_for_codigo_intraday(str(c), y, base_freq=base_freq)
            if not isinstance(result, dict):
                continue

            best_model = result.get("best_model", None)
            df_fut = result.get("df_fut", None)

            if best_model is None or df_fut is None or df_fut.empty:
                continue

            # si best_model == ENSEMBLE, asegúrate que exista esa columna:
            if best_model == "ENSEMBLE":
                # lo tratamos como "pred_ENSEMBLE"
                # construimos un df temporal con "pred_ENSEMBLE" como si fuera modelo
                df_tmp = df_fut.copy()
                # exporta best mean/median usando la columna pred_ENSEMBLE
                df_best_mean = monthly_wide_best_model_from_future(str(c), "ENSEMBLE", df_tmp.rename(columns={"pred_ENSEMBLE":"pred_ENSEMBLE"}), agg="mean")
                df_best_median = monthly_wide_best_model_from_future(str(c), "ENSEMBLE", df_tmp.rename(columns={"pred_ENSEMBLE":"pred_ENSEMBLE"}), agg="median")
            else:
                df_best_mean = monthly_wide_best_model_from_future(str(c), best_model, df_fut, agg="mean")
                df_best_median = monthly_wide_best_model_from_future(str(c), best_model, df_fut, agg="median")

            if not df_best_mean.empty:
                all_best_mean.append(df_best_mean)
            if not df_best_median.empty:
                all_best_median.append(df_best_median)

        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            continue

    # =========================================================
    # GUARDAR CSV GLOBAL (todos los códigos) usando mejor modelo
    # =========================================================
    if all_best_mean:
        df_all_mean = pd.concat(all_best_mean, ignore_index=True)
        out_all_mean = os.path.join(OUT_DIR, "ALL_best_model_monthly_MEAN_wide.csv")
        df_all_mean.to_csv(out_all_mean, index=False, encoding="utf-8")
        print(f"[OK] guardado GLOBAL best-model MEAN: {out_all_mean}")
    else:
        print("[WARN] No se generó GLOBAL best-model MEAN (sin resultados)")

    if all_best_median:
        df_all_median = pd.concat(all_best_median, ignore_index=True)
        out_all_median = os.path.join(OUT_DIR, "ALL_best_model_monthly_MEDIAN_wide.csv")
        df_all_median.to_csv(out_all_median, index=False, encoding="utf-8")
        print(f"[OK] guardado GLOBAL best-model MEDIAN: {out_all_median}")
    else:
        print("[WARN] No se generó GLOBAL best-model MEDIAN (sin resultados)")


if __name__ == "__main__":
    main()
