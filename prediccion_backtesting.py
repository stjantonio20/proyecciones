#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forecast_una_serie.py

Entrena y valida 2 enfoques:
  Opción 1: MIMO MLP (Dense(H)) sobre log1p(y) + lags + features calendario sin/cos
  Opción 2: Regresión con Fourier (K armónicos) + Ridge (tendencia + estacionalidad)

Valida con backtesting rolling-origin (expanding window) y luego entrena final para pronosticar FUTURE_H meses.
Genera gráfica: histórico + forecast + línea de fin histórico.

Uso:
  python forecast_una_serie.py --csv mi_serie.csv --future_h 13
  python forecast_una_serie.py --csv mi_serie.csv --future_h 13 --horizon 12 --test_windows 12

Formato CSV esperado (ancho):
  codigo,mar-15,abr-15,...,nov-25
  101101,18052.46,40869.87,...,3921940.12
o también sin columna codigo (solo una fila de valores con encabezados de meses).
python prediccion_backtesting.py --csv unaserie.csv --lookback 24 --horizon 12 --future_h 12 --test_windows 12
"""

import argparse
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# -----------------------------
# Utilidades de fechas / parsing
# -----------------------------

_ES_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12
}
_EN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def parse_month_col(col: str) -> pd.Timestamp:
    """
    Acepta 'mar-15', 'Mar-15', 'nov-2025', 'Nov-25', etc.
    Soporta meses español/inglés en 3 letras.
    """
    s = str(col).strip()
    s = s.replace("_", "-").replace("/", "-").lower()

    # ejemplo: mar-15, nov-2025
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"No pude parsear columna fecha: {col}")

    m_str, y_str = parts[0], parts[1]
    if m_str in _ES_MONTHS:
        m = _ES_MONTHS[m_str]
    elif m_str in _EN_MONTHS:
        m = _EN_MONTHS[m_str]
    else:
        raise ValueError(f"Mes no reconocido en columna: {col}")

    y = int(y_str)
    # si viene '15' => 2015 (asumimos 2000+)
    if y < 100:
        y += 2000

    return pd.Timestamp(year=y, month=m, day=1)

def coerce_float(x):
    """Convierte valores tipo '1,234,567.89' a float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(",", "")
    if s == "":
        return np.nan
    return float(s)

def load_wide_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    # Si hay columna 'codigo', la ignoramos para la serie
    cols = [c for c in df.columns if str(c).lower() != "codigo"]
    if len(df) == 0:
        raise ValueError("CSV vacío")
    row = df.iloc[0][cols].to_dict()

    # parsear columnas a fechas
    idx = []
    vals = []
    for c, v in row.items():
        try:
            ts = parse_month_col(c)
        except Exception:
            # si no parece fecha, ignoramos
            continue
        idx.append(ts)
        vals.append(coerce_float(v))

    s = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()
    s.name = "y"
    return s

# -----------------------------
# Features
# -----------------------------

def calendar_features(dti: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos de mes (12) + (opcional) sin/cos anual continuo"""
    month = dti.month.values.astype(float)
    # Estacionalidad mensual
    sin_m = np.sin(2 * np.pi * month / 12.0)
    cos_m = np.cos(2 * np.pi * month / 12.0)

    # Tendencia suave anual (tiempo continuo)
    t = np.arange(len(dti), dtype=float)
    sin_t = np.sin(2 * np.pi * t / 12.0)   # otra base, por si ayuda
    cos_t = np.cos(2 * np.pi * t / 12.0)

    return np.column_stack([sin_m, cos_m, sin_t, cos_t])

def make_supervised_log(y: pd.Series, lookback: int, horizon: int):
    """
    Construye dataset supervisado para MIMO:
      X = [log1p(y_{t-lookback:t-1}), calendar_features(t), ...]
      Y = [log1p(y_{t:t+horizon-1})]
    """
    y = y.copy().astype(float)
    y_log = np.log1p(y.values)

    idx = y.index
    cal = calendar_features(idx)

    X_list, Y_list, t_index = [], [], []
    for t in range(lookback, len(y) - horizon + 1):
        x_lags = y_log[t - lookback:t]
        x_cal = cal[t]  # features del tiempo t (inicio del forecast)
        X = np.concatenate([x_lags, x_cal], axis=0)
        Y = y_log[t:t + horizon]
        X_list.append(X)
        Y_list.append(Y)
        t_index.append(idx[t])

    X = np.asarray(X_list)
    Y = np.asarray(Y_list)
    return X, Y, pd.DatetimeIndex(t_index)

# -----------------------------
# Modelos
# -----------------------------

def build_model_option1_mimo_mlp(horizon: int, random_state: int = 24):
    """
    Opción 1: MIMO MLP en log-space.
    MultiOutputRegressor para predecir vector horizonte.
    """
    base = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.2,
    )
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mimo", MultiOutputRegressor(base))
    ])
    return model

def fourier_design(dti: pd.DatetimeIndex, K: int):
    """
    Matriz de diseño Fourier con K armónicos anuales para datos mensuales.
    Incluye intercepto + tendencia lineal + sin/cos.
    """
    n = len(dti)
    t = np.arange(n, dtype=float)
    X = [np.ones(n), t]

    # armónicos: k=1..K
    for k in range(1, K + 1):
        X.append(np.sin(2 * np.pi * k * t / 12.0))
        X.append(np.cos(2 * np.pi * k * t / 12.0))

    return np.column_stack(X)

def fit_predict_option2_fourier_ridge(y: pd.Series, future_h: int, K: int = 3, alpha: float = 1.0):
    """
    Opción 2: Ridge sobre Fourier + tendencia (en log1p).
    Devuelve predicción future_h en escala original.
    """
    y_log = np.log1p(y.values.astype(float))
    X = fourier_design(y.index, K=K)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, y_log)

    # futuro
    idx_future = pd.date_range(start=y.index[-1] + pd.offsets.MonthBegin(1), periods=future_h, freq="MS")
    X_future = fourier_design(idx_future, K=K)  # ojo: aquí t reinicia; mejor: usar t absoluto
    # Arreglamos t absoluto para continuidad:
    n = len(y)
    t_future = np.arange(n, n + future_h, dtype=float)
    Xf = [np.ones(future_h), t_future]
    for k in range(1, K + 1):
        Xf.append(np.sin(2 * np.pi * k * t_future / 12.0))
        Xf.append(np.cos(2 * np.pi * k * t_future / 12.0))
    X_future = np.column_stack(Xf)

    yhat_log = reg.predict(X_future)
    yhat = np.expm1(yhat_log)
    return pd.Series(yhat, index=idx_future)

# -----------------------------
# Backtesting
# -----------------------------

@dataclass
class Metrics:
    rmse: float
    mae: float
    mape: float

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1e-9)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return Metrics(rmse=rmse, mae=mae, mape=mape)

def rolling_backtest_option1(y: pd.Series, lookback: int, horizon: int, test_windows: int):
    """
    Expanding window:
      - Hace test_windows folds
      - En cada fold: entrena hasta t, predice siguiente horizon
    """
    X, Y, t_index = make_supervised_log(y, lookback=lookback, horizon=horizon)

    # folds al final
    n_samples = len(X)
    folds = min(test_windows, n_samples)
    start_fold = n_samples - folds

    metrics = []
    preds_last = None
    last_cut = None

    for i in range(start_fold, n_samples):
        X_train, Y_train = X[:i], Y[:i]
        X_test, Y_test = X[i:i+1], Y[i:i+1]  # una predicción por fold

        model = build_model_option1_mimo_mlp(horizon=horizon, random_state=24)
        model.fit(X_train, Y_train)

        yhat_log = model.predict(X_test)[0]
        ytrue_log = Y_test[0]

        yhat = np.expm1(yhat_log)
        ytrue = np.expm1(ytrue_log)

        m = compute_metrics(ytrue, yhat)
        metrics.append(m)

        preds_last = yhat
        last_cut = t_index[i]  # inicio del horizonte predicho

    # promedio métricas
    rmse = float(np.mean([m.rmse for m in metrics]))
    mae = float(np.mean([m.mae for m in metrics]))
    mape = float(np.mean([m.mape for m in metrics]))
    return Metrics(rmse, mae, mape), (last_cut, preds_last)

def rolling_backtest_option2(y: pd.Series, horizon: int, test_windows: int, K: int = 3, alpha: float = 1.0):
    """
    Backtest para opción 2:
      en cada fold ajusta Fourier+Ridge hasta t y predice horizon.
    """
    n = len(y)
    folds = min(test_windows, max(1, n - horizon - 12))
    # ubicamos cortes al final
    cut_points = [n - horizon - folds + j for j in range(folds)]  # índices de corte

    metrics = []
    preds_last = None
    last_cut_date = None

    y_vals = y.values.astype(float)

    for cut in cut_points:
        y_tr = y.iloc[:cut]
        y_te = y.iloc[cut:cut+horizon]

        # fit y predict horizon
        yhat = fit_predict_option2_fourier_ridge(y_tr, future_h=horizon, K=K, alpha=alpha)
        # alinear
        yhat = yhat.iloc[:len(y_te)].values
        ytrue = y_te.values

        m = compute_metrics(ytrue, yhat)
        metrics.append(m)

        preds_last = yhat
        last_cut_date = y_te.index[0]

    rmse = float(np.mean([m.rmse for m in metrics]))
    mae = float(np.mean([m.mae for m in metrics]))
    mape = float(np.mean([m.mape for m in metrics]))
    return Metrics(rmse, mae, mape), (last_cut_date, preds_last)

# -----------------------------
# Forecast final + Plot
# -----------------------------

def fit_forecast_option1(y: pd.Series, lookback: int, horizon: int, future_h: int):
    # Entrena con todos los ejemplos posibles
    X, Y, _ = make_supervised_log(y, lookback=lookback, horizon=horizon)
    model = build_model_option1_mimo_mlp(horizon=horizon, random_state=24)
    model.fit(X, Y)

    # para pronosticar FUTURE_H, hacemos “roll” por bloques de horizon
    y_hist = y.values.astype(float).copy()
    idx_hist = y.index

    preds = []
    total = future_h
    cur_series = y_hist.copy()

    while total > 0:
        # último lookback
        cur_idx = pd.date_range(start=idx_hist[0], periods=len(cur_series), freq="MS")
        cur_y = pd.Series(cur_series, index=cur_idx)

        X_last, _, _ = make_supervised_log(cur_y, lookback=lookback, horizon=horizon)
        x = X_last[-1:].copy()

        yhat_log = model.predict(x)[0]
        yhat = np.expm1(yhat_log)

        take = min(total, horizon)
        preds.append(yhat[:take])
        cur_series = np.concatenate([cur_series, yhat[:take]])
        total -= take

    preds = np.concatenate(preds)
    idx_future = pd.date_range(start=y.index[-1] + pd.offsets.MonthBegin(1), periods=future_h, freq="MS")
    return pd.Series(preds, index=idx_future)

def plot_history_forecast(y: pd.Series, pred: pd.Series, title: str, fin_historico: pd.Timestamp, out_png: str = None):
    plt.figure(figsize=(16, 5))
    plt.plot(y.index, y.values, label="Serie original")
    plt.plot(pred.index, pred.values, "o--", label=f"Proyección ({len(pred)} meses)")
    plt.axvline(fin_historico, linestyle="--", label="Fin histórico")
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV ancho con columnas mar-15..nov-25 (y opcional 'codigo').")
    ap.add_argument("--lookback", type=int, default=24, help="Meses de historia para features (opción 1).")
    ap.add_argument("--horizon", type=int, default=12, help="Horizonte de validación (meses).")
    ap.add_argument("--future_h", type=int, default=13, help="Meses a pronosticar final.")
    ap.add_argument("--test_windows", type=int, default=12, help="Número de folds al final para backtesting.")
    ap.add_argument("--fourier_K", type=int, default=3, help="Armónicos Fourier (opción 2).")
    ap.add_argument("--fourier_alpha", type=float, default=1.0, help="Ridge alpha (opción 2).")
    ap.add_argument("--out_png", default="", help="Si quieres guardar PNG, pon ruta (opcional).")
    args = ap.parse_args()

    y = load_wide_csv(args.csv).asfreq("MS")
    y = y.sort_index()

    # -------- Validación --------
    m1, (cut1, _) = rolling_backtest_option1(
        y, lookback=args.lookback, horizon=args.horizon, test_windows=args.test_windows
    )
    m2, (cut2, _) = rolling_backtest_option2(
        y, horizon=args.horizon, test_windows=args.test_windows, K=args.fourier_K, alpha=args.fourier_alpha
    )

    print("\n=== VALIDACIÓN (rolling backtest) ===")
    print(f"Opción 1 (MIMO MLP log1p + calendario) | RMSE={m1.rmse:,.2f} | MAE={m1.mae:,.2f} | MAPE={m1.mape:,.2f}%")
    print(f"Opción 2 (Fourier+Ridge en log1p)       | RMSE={m2.rmse:,.2f} | MAE={m2.mae:,.2f} | MAPE={m2.mape:,.2f}%")

    # -------- Forecast final --------
    # Nota: para la opción 1, reusamos horizon como bloque interno; future_h puede ser diferente
    pred1 = fit_forecast_option1(y, lookback=args.lookback, horizon=args.horizon, future_h=args.future_h)
    pred2 = fit_predict_option2_fourier_ridge(y, future_h=args.future_h, K=args.fourier_K, alpha=args.fourier_alpha)

    # Elegimos “ganador” por RMSE (puedes cambiar criterio)
    best_name = "Opción 1 (MIMO MLP)" if m1.rmse <= m2.rmse else "Opción 2 (Fourier+Ridge)"
    best_pred = pred1 if m1.rmse <= m2.rmse else pred2

    print(f"\nModelo elegido por menor RMSE: {best_name}")

    title = f"Serie mensual + proyección | {best_name}"
    out_png = args.out_png.strip() or None
    plot_history_forecast(y, best_pred, title=title, fin_historico=y.index[-1], out_png=out_png)

if __name__ == "__main__":
    main()
