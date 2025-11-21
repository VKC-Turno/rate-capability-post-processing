"""Plotters that consume precomputed SoC data/knee summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .soc_data import get_soc_dataframe
from .soc_plot import plot_soc_curve

PathLike = Union[str, Path]


def plot_cell_socv_grid(
    cell_csv: PathLike,
    stats_csv: Optional[PathLike] = None,
    stats_df: Optional[pd.DataFrame] = None,
    include_rest: bool = False,
    show: bool = True,
    soc_data_dir: Optional[PathLike] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot a grid (cycles x step names) of SoC-V curves for a single cell."""

    cell_csv = Path(cell_csv)
    stats_path = Path(stats_csv) if stats_csv is not None else None

    if not cell_csv.exists():
        raise FileNotFoundError(f"Cell data not found: {cell_csv}")
    if stats_path is None and stats_df is None:
        raise ValueError("Either `stats_csv` or `stats_df` must be supplied.")
    if stats_path is not None and not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    df, cell_name = get_soc_dataframe(cell_csv, soc_data_dir)
    stats = pd.read_csv(stats_path) if stats_df is None else stats_df

    cell_stats = stats[stats["cell_name"] == cell_name]
    if cell_stats.empty:
        raise ValueError(f"No stats rows found for cell '{cell_name}'.")

    step_names = sorted(cell_stats["step name"].unique())
    if not include_rest:
        step_names = [name for name in step_names if str(name).lower() != "rest"]
    if not step_names:
        raise ValueError(
            f"No step names available for cell '{cell_name}' after filtering Rest."
        )

    cycles = sorted(cell_stats["cycle no"].unique())
    n_rows = len(cycles)
    n_cols = len(step_names)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for r, cycle in enumerate(cycles):
        for c, step_name in enumerate(step_names):
            ax = axes[r, c]
            subset_stats = cell_stats[
                (cell_stats["cycle no"] == cycle)
                & (cell_stats["step name"] == step_name)
            ]
            if subset_stats.empty:
                ax.axis("off")
                continue

            c_rates = subset_stats["max_c_rate"].values
            norm = plt.Normalize(vmin=c_rates.min(), vmax=c_rates.max())
            cmap = plt.cm.plasma

            for _, row in subset_stats.sort_values("step no").iterrows():
                step_no = int(row["step no"])
                c_rate = row["max_c_rate"]
                trace = df[
                    (df["cycle no"] == cycle) & (df["step no"] == step_no)
                ]
                if trace.empty:
                    continue

                color = cmap(norm(c_rate))
                plot_soc_curve(
                    ax=ax,
                    trace=trace,
                    step_name=step_name,
                    color=color,
                    label=f"Step {step_no} ({c_rate:.2f}C)",
                )

            if r == 0:
                ax.set_title(step_name)
            if c == 0:
                ax.set_ylabel(f"Cycle {cycle}\nVoltage (V)")
            else:
                ax.set_ylabel("Voltage (V)")

            ax.set_xlabel("State of Charge (%)")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(frameon=False, fontsize=8, loc="best")

            if step_name.strip().lower() == "cc_dchg":
                ax.set_xlim(100, 0)
            else:
                ax.set_xlim(0, 100)

    fig.suptitle(f"{cell_name} - SoC-V by Cycle and Step Name", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if show:
        plt.show()

    return fig, axes


def export_socv_grid_pdf(
    data_dir: PathLike,
    stats_csv: PathLike,
    output_pdf: PathLike,
    include_rest: bool = False,
    soc_data_dir: Optional[PathLike] = "/home/kcv/Desktop/Rate_Capability/results/data/soc_data",
) -> Path:
    """Generate a multi-page PDF where each page shows one cell's SoC-V grid."""

    data_dir = Path(data_dir)
    stats_csv = Path(stats_csv)
    output_pdf = Path(output_pdf)
    soc_data_dir = Path(soc_data_dir) if soc_data_dir is not None else None

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_csv}")

    csv_files = sorted(data_dir.glob("RD_RateCapability_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No cell CSV files found under {data_dir}")

    stats = pd.read_csv(stats_csv)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for csv_path in csv_files:
            fig, _ = plot_cell_socv_grid(
                cell_csv=csv_path,
                stats_df=stats,
                include_rest=include_rest,
                show=False,
                soc_data_dir=soc_data_dir,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_pdf


__all__ = ["plot_cell_socv_grid", "export_socv_grid_pdf"]
