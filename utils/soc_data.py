"""Data utilities for State-of-Charge processing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def load_cell_dataframe(cell_csv: PathLike) -> Tuple[pd.DataFrame, str]:
    """Load a raw cell CSV and return (DataFrame, cell_name)."""

    cell_path = Path(cell_csv)
    df = pd.read_csv(cell_path)
    if "record number" in df.columns:
        df = df.sort_values("record number").reset_index(drop=True)
    cell_name = cell_path.stem
    return df, cell_name


def compute_soc_series(df: pd.DataFrame) -> pd.Series:
    """Compute SoC (%) for every (cycle, step) slice."""

    required = {"cycle no", "step no", "step name", "capacity(ah)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for SoC calculation: {missing}")

    soc_values = np.zeros(len(df), dtype=float)
    for (_, _), group in df.groupby(["cycle no", "step no"]):
        idx = group.index.to_numpy()
        cap = group["capacity(ah)"].abs().values
        max_cap = cap.max()
        if max_cap == 0:
            soc = np.zeros_like(cap)
        else:
            soc = cap / max_cap

        step_name = str(group["step name"].iloc[0]).strip().lower()
        if step_name == "cc_dchg":
            soc = 1.0 - soc

        soc_values[idx] = soc * 100.0

    return pd.Series(soc_values, index=df.index, name="soc(%)")


def ensure_soc_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a soc(%) column to ``df`` if it is missing."""

    if "soc(%)" not in df.columns:
        df = df.copy()
        df["soc(%)"] = compute_soc_series(df)
    return df


def save_soc_data(cell_csv: PathLike, output_dir: PathLike) -> Tuple[pd.DataFrame, Path]:
    """Compute SoC for a cell and persist the trimmed dataset."""

    df, cell_name = load_cell_dataframe(cell_csv)
    df = ensure_soc_column(df)
    df["cell_name"] = cell_name

    base_cols = [
        "cell_name",
        "cycle no",
        "step no",
        "step name",
        "current(a)",
        "volt(v)",
        "capacity(ah)",
    ]
    energy_cols = [col for col in df.columns if "energy" in col.lower()]
    cols = [col for col in base_cols if col in df.columns]
    cols.extend([col for col in energy_cols if col not in cols])
    cols.append("soc(%)")

    out_df = df.loc[:, cols]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cell_name}_soc_data.csv"
    out_df.to_csv(output_path, index=False)
    return df, output_path


def get_soc_dataframe(
    cell_csv: PathLike,
    output_dir: Optional[PathLike] = None,
) -> Tuple[pd.DataFrame, str]:
    """Return a SoC-augmented dataframe, optionally loading/saving from disk."""

    cell_path = Path(cell_csv)
    cell_name = cell_path.stem

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        soc_file = output_dir / f"{cell_name}_soc_data.csv"
        if soc_file.exists():
            df = pd.read_csv(soc_file)
            return df, cell_name
        df, _ = save_soc_data(cell_path, output_dir)
        return df, cell_name

    df, _ = load_cell_dataframe(cell_path)
    df = ensure_soc_column(df)
    return df, cell_name


__all__ = [
    "load_cell_dataframe",
    "compute_soc_series",
    "ensure_soc_column",
    "save_soc_data",
    "get_soc_dataframe",
]
