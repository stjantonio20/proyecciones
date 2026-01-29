#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import shutil
from pathlib import Path


ORIG_PLOT_PREFIX = "plot_forecast_only_"
ORIG_EXT = ".png"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_codigo(fname: str) -> str | None:
    """
    Extrae <codigo> desde plot_forecast_only_<codigo>.png
    """
    m = re.match(rf"^{ORIG_PLOT_PREFIX}(.+){ORIG_EXT}$", fname)
    return m.group(1) if m else None


def next_available(dst: Path) -> Path:
    """
    Evita sobreescritura: codigo_123.png, codigo_123_1.png, ...
    """
    if not dst.exists():
        return dst

    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        candidate = dst.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def copy_and_rename(src_root: Path, dst_dir: Path) -> int:
    if not src_root.exists():
        print(f"[WARN] No existe: {src_root}")
        return 0

    ensure_dir(dst_dir)
    copied = 0

    for img in src_root.rglob(f"{ORIG_PLOT_PREFIX}*{ORIG_EXT}"):
        # solo aceptar rutas tipo .../codigo_XXXX/plot_....
        if not any(p.startswith("codigo_") for p in img.parts):
            continue

        codigo = extract_codigo(img.name)
        if codigo is None:
            continue

        dst_name = f"codigo_{codigo}.png"
        dst_path = next_available(dst_dir / dst_name)

        shutil.copy2(img, dst_path)
        copied += 1
        print(f"[OK] {img} -> {dst_path}")

    return copied


def main():
    raiz = Path.cwd()

    # Orígenes
    src_36m = raiz / "proyeccion_13meses"
    src_1y  = raiz / "proyeccion_37meses"

    # Destinos
    base_dst = raiz / "proyeccion_13mesesy37meses"
    dst_3y = base_dst / "proyeccion_3anios"
    dst_1y_dst = base_dst / "proyeccion_1anio"

    ensure_dir(dst_3y)
    ensure_dir(dst_1y_dst)

    print("[INFO] Copiando modelos 36 meses → proyeccion_3anios")
    n36 = copy_and_rename(src_36m, dst_3y)

    print("\n[INFO] Copiando modelos 12 meses → proyeccion_1anio")
    n12 = copy_and_rename(src_1y, dst_1y_dst)

    print("\n[RESUMEN]")
    print(f"  3 años: {n36}")
    print(f"  1 año:  {n12}")
    print(f"  TOTAL:  {n36 + n12}")


if __name__ == "__main__":
    main()
