import pandas as pd
import numpy as np

df = pd.read_csv("Crediguate_rampa_nuevo_diario_30x.csv")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["codigo"] = df["codigo"].astype(str)

codigo = "101101"
g = df[df["codigo"] == codigo].sort_values("fecha")
idx = pd.DatetimeIndex(g["fecha"]).dropna()

# deltas en minutos
dmins = np.diff(idx.view("i8")) / 1e9 / 60
dmins = dmins[np.isfinite(dmins)]

print("n=", len(idx))
print("min delta (min):", np.min(dmins))
print("median delta (min):", np.median(dmins))
print("max delta (min):", np.max(dmins))

# top 10 deltas más comunes
vals, counts = np.unique(dmins.astype(int), return_counts=True)
top = sorted(zip(counts, vals), reverse=True)[:10]
print("Top deltas (count, minutes):", top)
