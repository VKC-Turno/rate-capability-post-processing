"""Utilities for generating state-of-charge vs. voltage (SoC-V) plots."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _knee_points(
    soc: np.ndarray, volt: np.ndarray, step_name: str
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return knee points by scanning user-defined SoC/voltage windows."""

    def _find_peak(bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        soc_bounds, volt_bounds = bounds
        mask = (
            (soc >= soc_bounds[0])
            & (soc <= soc_bounds[1])
            & (volt >= volt_bounds[0])
            & (volt <= volt_bounds[1])
        )
        if mask.sum() < 2:
            return float("nan"), float("nan")

        s = soc[mask]
        v = volt[mask]
        order = np.argsort(s)
        s = s[order]
        v = v[order]

        dv = np.gradient(v, s, edge_order=2)
        d2v = np.gradient(dv, s, edge_order=2)
        # large curvature implies sharp bump; also include slope change
        score = np.abs(d2v) * 0.7 + np.abs(dv) * 0.3
        idx = int(np.argmax(score))
        return float(s[idx]), float(v[idx])

    step_clean = step_name.strip().lower()
    if step_clean == "cccv_chg":
        windows = [
            ((0.0, 20.0), (3.2, 3.4)),
            ((80.0, 100.0), (3.4, 3.5)),
        ]
    elif step_clean == "cc_dchg":
        windows = [
            ((80.0, 100.0), (3.2, 3.4)),
            ((0.0, 30.0), (3.4, 3.5)),
        ]
    else:
        windows = [
            ((0.0, 20.0), (volt.min(), volt.max())),
            ((80.0, 100.0), (volt.min(), volt.max())),
        ]

    knees = [_find_peak(w) for w in windows]

    for i, knee in enumerate(knees):
        if np.isnan(knee[0]):
            knees[i] = (float("nan"), float("nan"))

    return knees[0], knees[1]


def _prepare_soc_trace(
    subset: pd.DataFrame, step_name: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Return SoC (%) vs voltage arrays plus knee point info."""

    if "record number" in subset.columns:
        subset = subset.sort_values("record number")
    else:
        subset = subset.sort_index()

    cap = subset["capacity(ah)"].abs().values
    max_cap = cap.max() if cap.size else 0.0
    if max_cap == 0:
        soc = np.zeros_like(cap)
    else:
        soc = cap / max_cap

    step_clean = step_name.strip().lower()
    if step_clean == "cc_dchg":
        soc = 1.0 - soc

    soc *= 100.0
    order = np.argsort(soc)
    soc = soc[order]
    volt = subset["volt(v)"].values[order]
    low_knee, high_knee = _knee_points(soc, volt, step_name)
    return soc, volt, low_knee, high_knee


def plot_socv_by_step(
    cell_csv: PathLike,
    stats_csv: Optional[PathLike],
    step_no: int,
    cycle_numbers: Optional[Sequence[int]] = None,
    save_dir: Optional[PathLike] = None,
    save: bool = False,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    df: Optional[pd.DataFrame] = None,
    stats_df: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot SoC vs. voltage lines for a single step across cycles."""

    cell_csv = Path(cell_csv)
    stats_path = Path(stats_csv) if stats_csv is not None else None

    if not cell_csv.exists():
        raise FileNotFoundError(f"Cell data not found: {cell_csv}")
    if stats_path is None and stats_df is None:
        raise ValueError("Either `stats_csv` or `stats_df` must be supplied.")
    if stats_path is not None and not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    df = pd.read_csv(cell_csv) if df is None else df
    stats = pd.read_csv(stats_path) if stats_df is None else stats_df
    cell_name = cell_csv.stem

    cell_stats = stats[stats["cell_name"] == cell_name]
    if cell_stats.empty:
        raise ValueError(
            f"No stats rows found for cell '{cell_name}'. "
            "Ensure the aggregated stats CSV contains this cell."
        )

    target_stats = cell_stats[cell_stats["step no"] == step_no].copy()
    if target_stats.empty:
        raise ValueError(f"Step {step_no} not found in stats for cell '{cell_name}'.")

    if cycle_numbers is not None:
        cycle_set = set(cycle_numbers)
        target_stats = target_stats[target_stats["cycle no"].isin(cycle_set)]
        if target_stats.empty:
            raise ValueError(
                f"Requested cycles {sorted(cycle_set)} do not contain step {step_no} "
                f"for cell '{cell_name}'."
            )

    c_rate_map = {
        (int(row["cycle no"]), int(row["step no"])): row["max_c_rate"]
        for _, row in target_stats.iterrows()
        if pd.notna(row["max_c_rate"])
    }
    if not c_rate_map:
        raise ValueError(
            f"No valid max_c_rate values found for cell '{cell_name}' step {step_no}."
        )

    cycles = sorted({cycle for cycle, _ in c_rate_map.keys()})

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    c_rates = np.array(list(c_rate_map.values()), dtype=float)
    norm = plt.Normalize(vmin=c_rates.min(), vmax=c_rates.max())
    cmap = plt.cm.plasma

    plotted_any = False
    step_name_value = None
    for cycle in cycles:
        subset = df[(df["cycle no"] == cycle) & (df["step no"] == step_no)]
        if subset.empty:
            continue

        if step_name_value is None and not subset["step name"].empty:
            step_name_value = str(subset["step name"].iloc[0]).strip().lower()

        c_rate = c_rate_map.get((cycle, step_no))
        if c_rate is None or np.isnan(c_rate):
            continue

        soc, volt, low_knee, high_knee = _prepare_soc_trace(
            subset, subset["step name"].iloc[0]
        )
        ax.plot(
            soc,
            volt,
            label=f"Cycle {cycle} ({c_rate:.2f}C)",
            color=cmap(norm(c_rate)),
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError(
            f"No matching raw data segments for cell '{cell_name}', "
            f"step {step_no} and cycles {cycles}."
        )

    ax.set_xlabel("State of Charge (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{cell_name} - Step {step_no} SoC-V by C-rate")
    ax.grid(True, linestyle="--", alpha=0.3)

    if step_name_value == "cc_dchg":
        ax.set_xlim(100, 0)
    else:
        ax.set_xlim(0, 100)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Max C-rate")

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    if save:
        if save_dir is None:
            raise ValueError("`save_dir` must be provided when `save=True`.")
        output_path = Path(save_dir) / f"{cell_name}_step{step_no}_socv.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_cycle_steps_socv(
    cell_csv: PathLike,
    stats_csv: PathLike,
    cycle_no: int,
    step_numbers: Optional[Sequence[int]] = None,
    save_dir: Optional[PathLike] = None,
    save: bool = False,
    show: bool = True,
) -> List[Tuple[int, plt.Figure, plt.Axes]]:
    """Generate SoC-V plots for each requested step within a cycle."""

    cell_csv = Path(cell_csv)
    stats_csv = Path(stats_csv)

    if not cell_csv.exists():
        raise FileNotFoundError(f"Cell data not found: {cell_csv}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_csv}")

    df = pd.read_csv(cell_csv)
    stats = pd.read_csv(stats_csv)
    cell_name = cell_csv.stem

    cycle_stats = stats[
        (stats["cell_name"] == cell_name) & (stats["cycle no"] == cycle_no)
    ]
    if cycle_stats.empty:
        raise ValueError(
            f"No stats rows found for cell '{cell_name}' cycle {cycle_no}."
        )

    available_steps = sorted(cycle_stats["step no"].unique())
    if step_numbers is None:
        steps = available_steps
    else:
        invalid = sorted(set(step_numbers) - set(available_steps))
        if invalid:
            raise ValueError(
                f"Steps {invalid} are not present for cycle {cycle_no} "
                f"of cell '{cell_name}'."
            )
        steps = list(step_numbers)

    outputs: List[Tuple[int, plt.Figure, plt.Axes]] = []
    for step in steps:
        fig, ax = plot_socv_by_step(
            cell_csv=cell_csv,
            stats_csv=stats_csv,
            step_no=step,
            cycle_numbers=[cycle_no],
            save_dir=save_dir,
            save=save,
            show=show,
            df=df,
            stats_df=stats,
        )
        outputs.append((step, fig, ax))

    return outputs


def plot_cell_socv_grid(
    cell_csv: PathLike,
    stats_csv: Optional[PathLike] = None,
    stats_df: Optional[pd.DataFrame] = None,
    include_rest: bool = False,
    show: bool = True,
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

    df = pd.read_csv(cell_csv)
    stats = pd.read_csv(stats_path) if stats_df is None else stats_df
    cell_name = cell_csv.stem

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

                soc, volt, low_knee, high_knee = _prepare_soc_trace(
                    trace, step_name
                )
                color = cmap(norm(c_rate))
                ax.plot(
                    soc,
                    volt,
                    label=f"Step {step_no} ({c_rate:.2f}C)",
                    color=color,
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
) -> Path:
    """Generate a multi-page PDF where each page shows one cell's SoC-V grid."""

    data_dir = Path(data_dir)
    stats_csv = Path(stats_csv)
    output_pdf = Path(output_pdf)

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
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_pdf


__all__ = [
    "plot_socv_by_step",
    "plot_cycle_steps_socv",
    "plot_cell_socv_grid",
    "export_socv_grid_pdf",
]
