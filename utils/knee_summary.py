"""Utilities for computing SoC knees and writing summaries."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .soc_data import get_soc_dataframe
from .soc_plot import piecewise_knees

PathLike = Union[str, Path]


def _process_cell(
    csv_path: Path,
    stats: pd.DataFrame,
    soc_data_dir: Optional[Path],
    allowed_cells: Optional[Sequence[str]],
    include_rest: bool,
    segments: int,
    low_index: int,
    high_index: int,
) -> List[dict]:
    cell_name = csv_path.stem
    if allowed_cells is not None and cell_name not in allowed_cells:
        return []

    df, cell_name = get_soc_dataframe(csv_path, soc_data_dir)
    cell_stats = stats[stats["cell_name"] == cell_name]
    if cell_stats.empty:
        return []

    stat_lookup = {
        (int(row["cycle no"]), int(row["step no"])): (
            row["step name"],
            row.get("max_c_rate"),
        )
        for _, row in cell_stats.iterrows()
    }

    rows: List[dict] = []
    grouped = df.groupby(["cycle no", "step no"], sort=False)
    for (cycle, step_no), trace in grouped:
        key = (int(cycle), int(step_no))
        if key not in stat_lookup:
            continue

        step_name, c_rate = stat_lookup[key]
        if not include_rest and str(step_name).strip().lower() == "rest":
            continue

        low_knee, high_knee = piecewise_knees(
            trace,
            step_name=step_name,
            segments=segments,
            low_index=low_index,
            high_index=high_index,
        )

        rows.append(
            {
                "cell_name": cell_name,
                "cycle_no": int(cycle),
                "step_no": int(step_no),
                "step_name": step_name,
                "max_c_rate": c_rate,
                "low_knee_soc": low_knee[0],
                "low_knee_voltage": low_knee[1],
                "high_knee_soc": high_knee[0],
                "high_knee_voltage": high_knee[1],
            }
        )

    return rows


def save_knee_summary(
    data_dir: PathLike,
    stats_csv: PathLike,
    output_csv: PathLike = "/home/kcv/Desktop/Rate_Capability/results/data/soc_knee_summary.csv",
    soc_data_dir: Optional[PathLike] = "/home/kcv/Desktop/Rate_Capability/results/data/soc_data",
    cell_names: Optional[Iterable[str]] = None,
    include_rest: bool = False,
    segments: int = 7,
    low_index: int = 2,
    high_index: int = 6,
    processes: Optional[int] = None,
) -> Path:
    """Precompute SoC knees for every (cycle, step) and persist them to CSV."""

    data_dir = Path(data_dir)
    stats_csv = Path(stats_csv)
    output_csv = Path(output_csv)
    soc_data_dir = Path(soc_data_dir) if soc_data_dir is not None else None

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_csv}")

    stats = pd.read_csv(stats_csv)
    allowed = {name.strip() for name in cell_names} if cell_names else None

    csv_paths = sorted(data_dir.glob("RD_RateCapability_*.csv"))
    if allowed:
        csv_paths = [p for p in csv_paths if p.stem in allowed]
    if not csv_paths:
        return output_csv

    worker_count = processes if processes is not None else cpu_count()
    worker_count = max(1, worker_count)
    tasks = [
        (
            path,
            stats,
            soc_data_dir,
            allowed,
            include_rest,
            segments,
            low_index,
            high_index,
        )
        for path in csv_paths
    ]

    rows: List[dict] = []
    if worker_count <= 1:
        for task in tasks:
            rows.extend(_process_cell(*task))
    else:
        with Pool(worker_count) as pool:
            for chunk in pool.starmap(_process_cell, tasks):
                rows.extend(chunk)

    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


__all__ = ["save_knee_summary"]
