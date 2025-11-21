"""Compute voltage sag and apparent direct resistance summaries from OCV datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

PathLike = str | Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OCV_DIR = REPO_ROOT / "results" / "data" / "ocv_power_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "data" / "adr_data"


def compute_adr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with voltage sag, apparent resistance, and voltage loss ratio."""

    required_cols = [
        "cycle no",
        "step no",
        "step name",
        "c_rate",
        "current(a)",
        "volt(v)",
        "OCV",
        "soc(%)",
        "resistance(ohm)",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in OCV data: {missing}")

    df = df.copy()
    volt = pd.to_numeric(df["volt(v)"], errors="coerce").to_numpy(float)
    ocv = pd.to_numeric(df["OCV"], errors="coerce").to_numpy(float)
    current = pd.to_numeric(df["current(a)"], errors="coerce").to_numpy(float)

    voltage_sag = ocv - volt
    with np.errstate(divide="ignore", invalid="ignore"):
        adr = np.where(np.abs(current) > 1e-12, voltage_sag / current, np.nan)
        vlr = np.where(np.abs(ocv) > 1e-12, voltage_sag / ocv, np.nan)

    df["voltage_sag(v)"] = voltage_sag
    df["apparent_direct_resistance(ohm)"] = adr
    df["voltage_loss_ratio"] = vlr
    columns = [
        col
        for col in [
            "absolute time",
            "cycle no",
            "step no",
            "step name",
            "c_rate",
            "current(a)",
            "volt(v)",
            "OCV",
            "soc(%)",
            "capacity(ah)",
            "energy(wh)",
            "charging energy(wh)",
            "discharge energy(wh)",
            "resistance(ohm)",
            "voltage_sag(v)",
            "apparent_direct_resistance(ohm)",
            "voltage_loss_ratio",
        ]
        if col in df.columns
    ]
    return df[
        columns
    ]


def save_cell_adr_data(ocv_csv: PathLike, output_dir: PathLike = DEFAULT_OUTPUT_DIR) -> Path:
    ocv_csv = Path(ocv_csv)
    if not ocv_csv.exists():
        raise FileNotFoundError(f"OCV data not found: {ocv_csv}")
    df = pd.read_csv(ocv_csv)
    summary = compute_adr_columns(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / ocv_csv.name.replace("_ocv_data", "_adr_data")
    summary.to_csv(out_path, index=False)
    return out_path


def process_directory(
    ocv_dir: PathLike = DEFAULT_OCV_DIR,
    output_dir: PathLike = DEFAULT_OUTPUT_DIR,
    glob: str = "*_ocv_data.csv",
) -> List[Path]:
    ocv_dir = Path(ocv_dir)
    files = sorted(ocv_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No OCV data files matching '{glob}' in {ocv_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Path] = []
    for csv_path in files:
        out = save_cell_adr_data(csv_path, output_dir=output_dir)
        results.append(out)
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate voltage-sag / ADR summaries for each cell.")
    parser.add_argument("--ocv-dir", type=Path, default=DEFAULT_OCV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glob", default="*_ocv_data.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    written = process_directory(args.ocv_dir, args.output_dir, args.glob)
    print(f"Wrote {len(written)} ADR datasets to {args.output_dir}")


if __name__ == "__main__":
    main()
