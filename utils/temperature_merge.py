"""Merge temperature TXT logs into per-cell CSV files and align with cell data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

COLS = ["arduino_no", "cell_name", "absolute time", "T_1", "T_2", "T_3", "T_4"]
MEAS_COLS = {
    "absolute time",
    "cycle no",
    "step no",
    "step name",
    "volt(v)",
    "current(a)",
    "capacity(ah)",
    "energy(wh)",
}


def _load_temperature_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS, engine="python")
    df["absolute time"] = pd.to_datetime(df["absolute time"], errors="coerce")
    df = df.dropna(subset=["absolute time"])
    df["source_file"] = path.name
    return df


def _load_cell_measurements(cell_id: str, cell_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if cell_dir is None:
        return None
    cell_dir = Path(cell_dir)
    candidates = [
        cell_dir / f"RD_RateCapability_{cell_id.zfill(4)}.csv",
        cell_dir / f"RD_RateCapability_{cell_id.zfill(3)}.csv",
    ]
    meas_path = next((p for p in candidates if p.exists()), None)
    if meas_path is None:
        return None
    df = pd.read_csv(meas_path, usecols=lambda c: c.lower() in MEAS_COLS, low_memory=False)
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    if "absolute time" not in df.columns:
        raise ValueError(f"'absolute time' column missing in {meas_path.name}")
    df["absolute time"] = pd.to_datetime(df["absolute time"], errors="coerce")
    df = df.dropna(subset=["absolute time"])
    keep_cols = [c for c in df.columns if c == "absolute time" or c in MEAS_COLS]
    return df[keep_cols].sort_values("absolute time")


def _merge_temp_and_cell(temp_df: pd.DataFrame, meas_df: Optional[pd.DataFrame], tolerance: str) -> pd.DataFrame:
    temp_df = temp_df.sort_values("absolute time")
    if meas_df is None:
        return temp_df
    tolerance_delta = pd.Timedelta(tolerance)
    merged = pd.merge_asof(
        temp_df,
        meas_df,
        on="absolute time",
        direction="nearest",
        tolerance=tolerance_delta,
    )
    return merged


def merge_temperature_directory(
    source_dir: Path,
    output_dir: Path,
    cell_data_dir: Optional[Path] = None,
    tolerance: str = "1s",
) -> List[Path]:
    """Merge temperature TXT files by cell and absolute time, optionally join with cell data."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_frames: Dict[str, pd.DataFrame] = {}
    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {source_dir}")

    for path in txt_files:
        df = _load_temperature_file(path)
        for cell_name, grp in df.groupby("cell_name"):
            merged = cell_frames.get(cell_name)
            if merged is None:
                cell_frames[cell_name] = grp.copy()
            else:
                combined = pd.concat([merged, grp], ignore_index=True)
                combined = combined.sort_values("absolute time").drop_duplicates("absolute time", keep="last")
                cell_frames[cell_name] = combined

    written_paths: List[Path] = []
    cell_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for cell_name, merged in cell_frames.items():
        merged = merged.sort_values("absolute time")
        digits = re.search(r"(\d{3,4})$", str(cell_name))
        meas_df = None
        if digits:
            cell_id = digits.group(1)
            if cell_id not in cell_cache:
                cell_cache[cell_id] = _load_cell_measurements(cell_id, cell_data_dir)
            meas_df = cell_cache[cell_id]

        merged = _merge_temp_and_cell(merged, meas_df, tolerance)
        out_path = output_dir / f"{cell_name}_temperature.csv"
        merged.drop(columns=["source_file"], errors="ignore").to_csv(out_path, index=False)
        written_paths.append(out_path)
    return written_paths


if __name__ == "__main__":
    SOURCE = Path("/home/kcv/Desktop/Rate_Capability/Data/temperature")
    OUTPUT = Path("/home/kcv/Desktop/Rate_Capability/results/data/merged_temperarure_data")
    CELL_DATA = Path("/home/kcv/Desktop/Rate_Capability/Data")
    paths = merge_temperature_directory(SOURCE, OUTPUT, cell_data_dir=CELL_DATA, tolerance="1s")
    print(f"Wrote {len(paths)} merged CSV files to {OUTPUT}")
