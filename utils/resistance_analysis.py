"""Helpers to compute simple delta-V/delta-I resistance summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .soc_data import load_cell_dataframe

PathLike = Union[str, Path]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "data" / "resistance_data"


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"cycle no", "step no", "step name", "volt(v)", "current(a)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for resistance calculation: {missing}")


def compute_resistance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-row delta-V/delta-I resistance estimates."""

    _ensure_required_columns(df)

    if "record number" in df.columns:
        df = df.sort_values("record number").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    groups = df.groupby(["cycle no", "step no", "step name"], dropna=False, sort=True, group_keys=False)
    rows = []
    for (cycle_no, step_no, step_name), subset in groups:
        if len(subset) < 2:
            continue
        volt = subset["volt(v)"].astype(float).to_numpy()
        curr = subset["current(a)"].astype(float).to_numpy()
        row_idx = subset.index.to_numpy()

        delta_v = np.diff(volt)
        delta_i = np.diff(curr)
        resistance = np.full_like(delta_v, np.nan, dtype=float)
        valid = np.abs(delta_i) > 1e-9
        resistance[valid] = delta_v[valid] / delta_i[valid]

        for idx, dv, di, r in zip(row_idx[1:], delta_v, delta_i, resistance):
            rows.append(
                {
                    "row_index": int(idx),
                    "cycle no": cycle_no,
                    "step no": step_no,
                    "step name": step_name,
                    "delta current(a)": float(di),
                    "delta volt(v)": float(dv),
                    "resistance(ohm)": float(r),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "row_index",
                "cycle no",
                "step no",
                "step name",
                "delta current(a)",
                "delta volt(v)",
                "resistance(ohm)",
            ]
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(["cycle no", "step no", "row_index"]).reset_index(drop=True)
    return result


def save_cell_resistance(cell_csv: PathLike, output_dir: PathLike) -> Path:
    """Compute resistance deltas for ``cell_csv`` and persist them."""

    df, cell_name = load_cell_dataframe(cell_csv)
    resistance_df = compute_resistance_table(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{cell_name}_resistance.csv"
    resistance_df.insert(0, "cell_name", cell_name)
    resistance_df.to_csv(out_file, index=False)
    return out_file


def process_cells(
    cell_paths: Sequence[PathLike],
    output_dir: PathLike = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Compute and save resistance data for an iterable of CSV paths."""

    written_paths = []
    for cell_csv in cell_paths:
        out_file = save_cell_resistance(cell_csv, output_dir)
        written_paths.append(out_file)
    return written_paths


def process_data_directory(
    data_dir: PathLike = DEFAULT_DATA_DIR,
    output_dir: PathLike = DEFAULT_OUTPUT_DIR,
    glob_pattern: str = "RD_RateCapability_*.csv",
) -> List[Path]:
    """Process every CSV in ``data_dir`` that matches ``glob_pattern``."""

    data_path = Path(data_dir)
    cell_paths = sorted(data_path.glob(glob_pattern))
    if not cell_paths:
        raise FileNotFoundError(f"No CSV files matching {glob_pattern} in {data_path}")
    return process_cells(cell_paths, output_dir=output_dir)


def _parse_args(argv: Iterable[str] | None = None) -> Tuple[Path, Path, str, Sequence[str]]:
    import argparse

    parser = argparse.ArgumentParser(description="Compute delta-V/delta-I resistance summaries per cell.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glob", default="RD_RateCapability_*.csv", help="Glob to match cell CSV files.")
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional explicit list of cell CSV files to process (overrides --glob).",
    )
    args = parser.parse_args(argv)

    if args.cells:
        cell_paths: Sequence[str] = args.cells
    else:
        cell_paths = ()

    return args.data_dir, args.output_dir, args.glob, cell_paths


def main(argv: Iterable[str] | None = None) -> None:
    data_dir, output_dir, glob_pattern, explicit_cells = _parse_args(argv)
    if explicit_cells:
        cell_paths = [Path(path) for path in explicit_cells]
    else:
        data_path = Path(data_dir)
        cell_paths = sorted(data_path.glob(glob_pattern))

    if not cell_paths:
        raise SystemExit("No cell CSV files found to process.")

    processed = process_cells(cell_paths, output_dir=output_dir)
    print(f"Wrote {len(processed)} resistance tables to {output_dir}")


if __name__ == "__main__":
    main()
