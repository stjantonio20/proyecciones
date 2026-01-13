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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"   # ayuda a que TF sea determinista (sobre todo en GPU, pero no estorba en CPU)

import random
random.seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "Crediguate_formato_mensual.csv"     # <-- tu archivo ancho: codigo | Mar-15 | ... | Nov-25
OUT_DIR  = "outputs_5modelos"       # <-- carpeta de salida
H_FUTURE = 12                       # meses a predecir

TRAIN_RATIO = 0.92                  # 98% train, 2% test
MIN_TEST_POINTS = 3                 # por si 2% da muy pocos, forzamos al menos 3 meses

# Si quieres correr SOLO un código:
#ONLY_CODIGO = None   # ejemplo: "101101"   o None para todos
#ONLY_CODIGO = "103101"   # ejemplo: "101101"   o None para todos
ONLY_CODIGO = "101101"   # ejemplo: "101101"   o None para todos

# Lookback para modelos supervisados (LSTM/TCN/ML/DL-multitarea)
LOOKBACK = 16        # 24 meses suele ir bien para mensual (2 años)

# ======= SWITCHES (comenta/descomenta) =======
RUN_ETS           = True   # (1) "literatura/comunidad": ETS Holt-Winters
RUN_TCN           = True   # (2) TCN
RUN_ML            = True   # (3) ML (LightGBM o fallback sklearn)
RUN_LSTM          = True   # (4) LSTM
RUN_MULTITASK_DL  = True   # (5) DL 2-cabezas: nivel + delta
# ============================================

# Neural nets: epochs/batch
NN_EPOCHS = 500 #100
NN_BATCH  = 36 #32
NN_PATIENCE = 20 #10


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


# =========================================================
# Scaling robusto (asinh + Standard)
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
# Windows supervisadas
# =========================================================
def make_supervised(y_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    y_scaled: (T,) -> X: (N, lookback, 1), y: (N, 1)
    """
    y_scaled = np.asarray(y_scaled, float).reshape(-1)
    X, Y = [], []
    for i in range(lookback, len(y_scaled)):
        X.append(y_scaled[i - lookback:i].reshape(lookback, 1))
        Y.append([y_scaled[i]])
    return np.array(X, float), np.array(Y, float)

def month_features(idx: pd.DatetimeIndex) -> np.ndarray:
    """
    Features estacionales: sin/cos del mes
    """
    m = idx.month.values.astype(float)
    sinm = np.sin(2*np.pi*m/12.0)
    cosm = np.cos(2*np.pi*m/12.0)
    return np.c_[sinm, cosm]


# =========================================================
# (1) ETS Holt-Winters (statsmodels)
# =========================================================
def fit_predict_ets(y_train: pd.Series, idx_all: pd.DatetimeIndex, test_len: int, h_future: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # si hay suficiente historia para estacionalidad 12
    seasonal = "add" if len(y_train) >= 36 else None
    trend = "add" if len(y_train) >= 24 else None

    model = ExponentialSmoothing(
        y_train.values,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)

    # predicción para test (últimos test_len) + futuro (h_future)
    # idx_all = train+test (observado)
    # forecast empieza justo después de train_end
    yhat_test = model.forecast(test_len)  # test inmediato
    yhat_fut  = model.forecast(test_len + h_future)[-h_future:]  # siguiente año

    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# (3) ML: LightGBM (fallback sklearn) con lags
# =========================================================
def fit_predict_ml(y: pd.Series, train_end_i: int, lookback: int, test_len: int, h_future: int):
    """
    Entrena con lags (en escala real, pero podrías cambiarlo a asinh)
    y predice test y futuro autoregresivo.
    """
    y_vals = y.values.astype(float)
    idx = y.index

    # Construir dataset supervisado con lags
    def build_XY(y_arr, idx_arr):
        X, Y, X_idx = [], [], []
        for t in range(lookback, len(y_arr)):
            lags = y_arr[t-lookback:t]
            feats = np.r_[lags, month_features(pd.DatetimeIndex([idx_arr[t]])).ravel()]
            X.append(feats)
            Y.append(y_arr[t])
            X_idx.append(idx_arr[t])
        return np.array(X, float), np.array(Y, float), pd.DatetimeIndex(X_idx)

    X, Y, X_idx = build_XY(y_vals, idx)

    # determinar rangos train/test en el dataset supervisado
    # train_end_i es índice en y (serie original) donde termina train (exclusivo)
    # en X_idx, el punto t corresponde a y[t]
    train_mask = (X_idx < idx[train_end_i])
    test_mask  = (X_idx >= idx[train_end_i])  # test

    Xtr, Ytr = X[train_mask], Y[train_mask]
    Xte, Yte = X[test_mask], Y[test_mask]
    # recortar a test_len por seguridad
    Xte, Yte = Xte[:test_len], Yte[:test_len]

    if len(Xtr) < 20:
        raise ValueError("Muy pocos datos para ML con lags (train < 20).")

    # modelo
    model = None
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(random_state=42)

    model.fit(Xtr, Ytr)

    # pred test directo (one-step, con lags reales)
    yhat_test = model.predict(Xte)

    # forecast futuro autoregresivo
    last_known = y_vals.copy()
    last_idx = idx[-1]

    # si el dataset termina en Nov-25, forecast arranca en Dec-25
    fut_idx = pd.date_range(start=last_idx + pd.offsets.MonthBegin(1), periods=h_future, freq="MS")

    yhat_fut = []
    y_buffer = last_known.tolist()
    for ts in fut_idx:
        lags = np.array(y_buffer[-lookback:], float)
        feats = np.r_[lags, month_features(pd.DatetimeIndex([ts])).ravel()]
        y_next = float(model.predict(feats.reshape(1, -1))[0])
        yhat_fut.append(y_next)
        y_buffer.append(y_next)

    return np.asarray(yhat_test, float), np.asarray(yhat_fut, float)


# =========================================================
# Keras builders: TCN / LSTM / MultiTask DL
# =========================================================
def get_tf():
    import tensorflow as tf
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
        # instala keras-tcn si falta
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
    """
    Modelo "multi-cabeza":
      - head1: predice nivel y(t)
      - head2: predice delta (y(t)-y(t-1))
    """
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
    tf = get_tf()
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
        # Ytr: (N,1) -> delta usando diferencia en escala
        # delta(t) = y(t) - y(t-1)  en escala transformada
        # Para cada sample i, y corresponde a y[t], y_prev corresponde al último lag
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
        # multitask: yhat[0] es nivel
        return np.asarray(yhat[0]).reshape(-1)
    return np.asarray(yhat).reshape(-1)

def forecast_keras_autoregressive(model, last_window_scaled: np.ndarray, h_future: int, multitask=False):
    """
    last_window_scaled: shape (lookback,) en escala transformada
    devuelve: (h_future,) en escala transformada
    """
    buf = last_window_scaled.astype(float).reshape(-1).tolist()
    preds = []
    for _ in range(h_future):
        x = np.array(buf[-len(last_window_scaled):], float).reshape(1, len(last_window_scaled), 1)
        yhat = model.predict(x, verbose=0)
        if isinstance(yhat, list):
            y_next = float(yhat[0].reshape(-1)[0])  # nivel
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

    # real
    plt.plot(y.index, y.values, label="Real", linewidth=2)

    # marca corte train/test
    plt.axvline(train_end, linestyle="--", linewidth=1)
    plt.text(train_end, np.nanmin(y.values), "  train_end", rotation=90, va="bottom")

    # test preds
    for name, p in preds_test.items():
        if p is None:
            continue
        m = min(len(test_idx), len(p))
        plt.plot(test_idx[:m], p[:m], label=f"{name} (test)")

    # future preds
    for name, p in preds_future.items():
        if p is None:
            continue
        m = min(len(future_idx), len(p))
        plt.plot(future_idx[:m], p[:m], label=f"{name} (future)", linestyle=":")

    plt.title(f"Codigo {codigo} — Test + Forecast {len(future_idx)}m (train={int(TRAIN_RATIO*100)}%)")
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
    # base frame con real
    df = pd.DataFrame({"fecha": y.index, "real": y.values})
    df = df.set_index("fecha")

    # columnas test y future
    for name, p in preds_test.items():
        if p is None:
            continue
        s = pd.Series(p, index=test_idx[:len(p)])
        df[f"pred_test_{name}"] = s

    for name, p in preds_future.items():
        if p is None:
            continue
        s = pd.Series(p, index=future_idx[:len(p)])
        df[f"pred_future_{name}"] = s

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
    # normaliza 'codigo'
    for c in df.columns:
        if c.lower() == "codigo" and c != "codigo":
            df = df.rename(columns={c: "codigo"})
            break
    if "codigo" not in df.columns:
        raise ValueError("No existe columna 'codigo' en el CSV.")

    df["codigo"] = df["codigo"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    month_cols = [c for c in df.columns if c != "codigo"]

    col_to_ts = {}
    for c in month_cols:
        col_to_ts[c] = parse_month_col(c)

    # melt a long
    long = df.melt(id_vars=["codigo"], value_vars=month_cols, var_name="mes", value_name="valor")
    long["fecha"] = long["mes"].map(col_to_ts)
    long["valor"] = pd.to_numeric(long["valor"], errors="coerce")

    long = long.dropna(subset=["fecha"]).sort_values(["codigo", "fecha"])
    return long


def series_by_codigo(long: pd.DataFrame, codigo: str) -> pd.Series:
    g = long[long["codigo"] == codigo].copy()
    s = pd.Series(g["valor"].values, index=pd.to_datetime(g["fecha"]))
    s = s.sort_index()
    # mensual MS
    s.index = s.index.to_period("M").to_timestamp(how="S")  # MonthStart compatible
    s = s.asfreq("MS")
    return s.astype(float)


# =========================================================
# MAIN RUN por codigo
# =========================================================
def run_for_codigo(codigo: str, s: pd.Series):
    s = s.dropna()
    s = s.astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < (LOOKBACK + 12):
        print(f"[SKIP] {codigo}: muy corta (n={len(s)})")
        return

    # split 98/2 con mínimo
    n = len(s)
    test_len = max(int(round(n * (1.0 - TRAIN_RATIO))), MIN_TEST_POINTS)
    test_len = min(test_len, max(3, n - LOOKBACK - 1))  # evita quedar sin train suficiente
    train_end_i = n - test_len
    if train_end_i <= LOOKBACK + 5:
        print(f"[SKIP] {codigo}: train insuficiente con este split (n={n}, test={test_len})")
        return

    y_train = s.iloc[:train_end_i]
    y_test  = s.iloc[train_end_i:]
    train_end_date = s.index[train_end_i]  # primer mes de test (corte visual)

    test_idx = y_test.index
    future_idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(1), periods=H_FUTURE, freq="MS")

    # contenedores
    preds_test: Dict[str, Optional[np.ndarray]] = {}
    preds_fut:  Dict[str, Optional[np.ndarray]] = {}
    scores: Dict[str, float] = {}

    # ============ (1) ETS ============
    if RUN_ETS:
        try:
            yhat_test, yhat_fut = fit_predict_ets(y_train, s.index, test_len=len(y_test), h_future=H_FUTURE)
            preds_test["ETS"] = yhat_test
            preds_fut["ETS"]  = yhat_fut
            scores["ETS"] = rmse(y_test.values, yhat_test)
        except Exception as e:
            print(f"[WARN] {codigo} ETS falló: {e}")
            preds_test["ETS"] = None
            preds_fut["ETS"]  = None

    # Preparar escala para NNs
    sc = fit_asinh_scaler(y_train.values)
    y_scaled_all = transform_asinh(s.values, sc)

    # dataset supervisado
    X_all, Y_all = make_supervised(y_scaled_all, LOOKBACK)

    # para mapear indices: el target Y_all[k] corresponde a s.index[LOOKBACK + k]
    target_idx = s.index[LOOKBACK:]

    # train/test en espacio supervisado
    # train_end_i en serie original => targets con fecha < s.index[train_end_i]
    train_mask = target_idx < s.index[train_end_i]
    test_mask  = target_idx >= s.index[train_end_i]

    Xtr, Ytr = X_all[train_mask], Y_all[train_mask]
    Xte, Yte = X_all[test_mask],  Y_all[test_mask]

    # recortar test a test_len real
    Xte, Yte = Xte[:len(y_test)], Yte[:len(y_test)]

    # split train/val (último 10% como validación)
    ntr = len(Xtr)
    nval = max(10, int(0.1 * ntr))
    Xtrain, Ytrain = Xtr[:-nval], Ytr[:-nval]
    Xval,   Yval   = Xtr[-nval:], Ytr[-nval:]

    # ventana final para forecast
    last_window = y_scaled_all[-LOOKBACK:]  # (lookback,)


    # ============ (2) TCN ============
    if RUN_TCN:
        try:
            m_tcn = build_tcn(LOOKBACK)
            train_keras_model(m_tcn, Xtrain, Ytrain, Xval, Yval, multitask=False)

            yhat_test_scaled = predict_keras_one_step(m_tcn, Xte)
            yhat_test = inverse_asinh(yhat_test_scaled, sc)

            yhat_fut_scaled = forecast_keras_autoregressive(m_tcn, last_window, H_FUTURE, multitask=False)
            yhat_fut = inverse_asinh(yhat_fut_scaled, sc)

            preds_test["TCN"] = yhat_test
            preds_fut["TCN"]  = yhat_fut
            scores["TCN"] = rmse(y_test.values, yhat_test)
        except Exception as e:
            print(f"[WARN] {codigo} TCN falló: {e}")
            preds_test["TCN"] = None
            preds_fut["TCN"]  = None

    # ============ (4) LSTM ============
    if RUN_LSTM:
        try:
            m_lstm = build_lstm(LOOKBACK)
            train_keras_model(m_lstm, Xtrain, Ytrain, Xval, Yval, multitask=False)

            yhat_test_scaled = predict_keras_one_step(m_lstm, Xte)
            yhat_test = inverse_asinh(yhat_test_scaled, sc)

            yhat_fut_scaled = forecast_keras_autoregressive(m_lstm, last_window, H_FUTURE, multitask=False)
            yhat_fut = inverse_asinh(yhat_fut_scaled, sc)

            preds_test["LSTM"] = yhat_test
            preds_fut["LSTM"]  = yhat_fut
            scores["LSTM"] = rmse(y_test.values, yhat_test)
        except Exception as e:
            print(f"[WARN] {codigo} LSTM falló: {e}")
            preds_test["LSTM"] = None
            preds_fut["LSTM"]  = None

    # ============ (5) MultiTask DL ============
    if RUN_MULTITASK_DL:
        try:
            m_mt = build_multitask_dl(LOOKBACK)
            train_keras_model(m_mt, Xtrain, Ytrain, Xval, Yval, multitask=True)

            yhat_test_scaled = predict_keras_one_step(m_mt, Xte)   # usa head nivel
            yhat_test = inverse_asinh(yhat_test_scaled, sc)

            yhat_fut_scaled = forecast_keras_autoregressive(m_mt, last_window, H_FUTURE, multitask=True)
            yhat_fut = inverse_asinh(yhat_fut_scaled, sc)

            preds_test["DL_MultiTask"] = yhat_test
            preds_fut["DL_MultiTask"]  = yhat_fut
            scores["DL_MultiTask"] = rmse(y_test.values, yhat_test)
        except Exception as e:
            print(f"[WARN] {codigo} MultiTask DL falló: {e}")
            preds_test["DL_MultiTask"] = None
            preds_fut["DL_MultiTask"]  = None

    # ============ (3) ML ============
    if RUN_ML:
        try:
            yhat_test, yhat_fut = fit_predict_ml(
                s, train_end_i=train_end_i, lookback=LOOKBACK, test_len=len(y_test), h_future=H_FUTURE
            )
            preds_test["ML"] = yhat_test
            preds_fut["ML"]  = yhat_fut
            scores["ML"] = rmse(y_test.values, yhat_test)
        except Exception as e:
            print(f"[WARN] {codigo} ML falló: {e}")
            preds_test["ML"] = None
            preds_fut["ML"]  = None

    # ===== imprimir ranking =====
    scored = sorted(scores.items(), key=lambda kv: kv[1])
    print(f"\n=== {codigo} (n={len(s)}, test={len(y_test)}) ===")
    for name, r in scored:
        print(f"  {name:12s} RMSE_test = {r:,.4f}")

    # ===== outputs =====
    out_c_dir = os.path.join(OUT_DIR, f"codigo_{codigo}")
    ensure_dir(out_c_dir)

    out_png = os.path.join(out_c_dir, f"plot_{codigo}.png")
    out_csv = os.path.join(out_c_dir, f"pred_{codigo}.csv")

    plot_all_models(
        codigo=codigo,
        y=s,
        train_end=train_end_date,
        test_idx=test_idx,
        future_idx=future_idx,
        preds_test=preds_test,
        preds_future=preds_fut,
        out_png=out_png
    )

    export_csv_codigo(
        out_csv=out_csv,
        y=s,
        test_idx=test_idx,
        future_idx=future_idx,
        preds_test=preds_test,
        preds_future=preds_fut
    )

    print(f"[OK] guardado: {out_png}")
    print(f"[OK] guardado: {out_csv}")


def main():
    ensure_dir(OUT_DIR)
    long = read_wide_monthly(CSV_PATH)

    codigos = sorted(long["codigo"].unique().tolist())
    if ONLY_CODIGO is not None:
        codigos = [str(ONLY_CODIGO)]

    print(f"[INFO] codigos a procesar: {len(codigos)}")
    print(f"[INFO] outputs: {OUT_DIR}")

    for c in codigos:
        try:
            s = series_by_codigo(long, str(c))
            # si toda la serie es NaN / vacía
            if s.dropna().empty:
                print(f"[SKIP] {c}: serie vacía")
                continue
            run_for_codigo(str(c), s)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            continue


if __name__ == "__main__":
    main()
