#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_por_anio.py

Grafica la serie en el tiempo, segmentando por año (un color distinto por año).

Uso:
  python plot_por_anio.py --csv mi_serie.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ES_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12
}
_EN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def parse_month_col(col: str) -> pd.Timestamp:
    s = str(col).strip().replace("_", "-").replace("/", "-").lower()
    m_str, y_str = s.split("-")
    if m_str in _ES_MONTHS:
        m = _ES_MONTHS[m_str]
    elif m_str in _EN_MONTHS:
        m = _EN_MONTHS[m_str]
    else:
        raise ValueError(f"Mes no reconocido: {col}")
    y = int(y_str)
    if y < 100:
        y += 2000
    return pd.Timestamp(year=y, month=m, day=1)

def coerce_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s == "":
        return np.nan
    return float(s)

def load_wide_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if str(c).lower() != "codigo"]
    row = df.iloc[0][cols].to_dict()

    idx, vals = [], []
    for c, v in row.items():
        try:
            ts = parse_month_col(c)
        except Exception:
            continue
        idx.append(ts)
        vals.append(coerce_float(v))

    s = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()
    s.name = "y"
    return s.asfreq("MS")

def plot_one_color_per_year(y: pd.Series, title: str = "Serie (un color por año)"):
    plt.figure(figsize=(16, 5))

    years = sorted(y.index.year.unique())
    for yr in years:
        seg = y[y.index.year == yr]
        plt.plot(seg.index, seg.values, label=str(yr))

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=min(6, len(years)))
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    y = load_wide_csv(args.csv)
    plot_one_color_per_year(y)

if __name__ == "__main__":
    main()
