"""ICA helper: generate dQ/dV plots and peak summaries for every cell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks, savgol_filter

PathLike = str | Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADR_DIR = REPO_ROOT / "results" / "data" / "adr_data"
DEFAULT_OUTPUT_PDF = REPO_ROOT / "results" / "plots" / "ica_report.pdf"
DEFAULT_PEAKS_CSV = REPO_ROOT / "results" / "data" / "ica_peak_summary.csv"
DEFAULT_PEAKS_TRENDS_PDF = REPO_ROOT / "results" / "plots" / "ica_peak_trends.pdf"
DEFAULT_STEP_ORDER = ("CC_DChg", "CCCV_Chg")
ALLOWED_RATES = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


def snap_c_rate(val: float) -> float:
    if pd.isna(val):
        return np.nan
    sign = np.sign(val) if val else 1.0
    target = ALLOWED_RATES[np.argmin(np.abs(ALLOWED_RATES - abs(val)))]
    return sign * target


def resample_trace(
    grp: pd.DataFrame, window: Tuple[float, float], dv: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate capacity/soc vs voltage inside ``window`` at spacing ``dv``."""

    vmin, vmax = window
    trimmed = grp[(grp["volt(v)"] >= vmin) & (grp["volt(v)"] <= vmax)].sort_values("volt(v)")
    if trimmed.empty:
        return None

    volt = trimmed["volt(v)"].to_numpy()
    cap = trimmed["capacity(ah)"].to_numpy()
    soc = trimmed["soc(%)"].to_numpy()

    dense_v = np.arange(volt.min(), volt.max(), dv)
    if dense_v.size < 3:
        return None

    dense_cap = np.interp(dense_v, volt, cap)
    dense_soc = np.interp(dense_v, volt, soc)
    return dense_v, dense_cap, dense_soc


def generate_ica_plots(
    adr_dir: PathLike = DEFAULT_ADR_DIR,
    output_pdf: PathLike = DEFAULT_OUTPUT_PDF,
    peaks_csv: PathLike = DEFAULT_PEAKS_CSV,
    dv: float | Dict[str, float] = 0.002,
    voltage_windows: Dict[str, Tuple[float, float]] | None = None,
    step_order: Sequence[str] = DEFAULT_STEP_ORDER,
    savgol_window: int | None = None,
    savgol_poly: int = 3,
    use_savgol_filter: bool = False,
    peaks_per_step: Dict[str, int] | None = None,
) -> Tuple[Path, Path]:
    """Render dQ/dV plots for every cell/cycle and save peak summary."""

    adr_dir = Path(adr_dir)
    output_pdf = Path(output_pdf)
    peaks_csv = Path(peaks_csv)

    if voltage_windows is None:
        voltage_windows = {
            "CC_DChg": (2.85, 3.55),
            "CCCV_Chg": (2.70, 3.65),
        }

    adr_files = sorted(adr_dir.glob("*_adr_data.csv"))
    if not adr_files:
        raise FileNotFoundError(f"No ADR files found in {adr_dir}")

    peaks_records: List[Dict[str, float]] = []
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    peaks_csv.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for adr_file in adr_files:
            df = pd.read_csv(adr_file)
            if df.empty:
                continue

            df["cycle no"] = pd.to_numeric(df["cycle no"], errors="coerce")
            df["c_rate_raw"] = pd.to_numeric(df["c_rate"], errors="coerce")
            # Remove ~0.33C before rounding
            skip_mask = np.isclose(df["c_rate_raw"].abs(), 0.33, atol=0.01)
            df = df[~skip_mask]
            df["c_rate"] = df["c_rate_raw"].apply(snap_c_rate)

            numeric_cols = ["volt(v)", "capacity(ah)", "soc(%)"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            mask = df["step name"].str.strip().isin(step_order)
            df = df[mask].dropna(subset=["volt(v)", "capacity(ah)", "soc(%)", "c_rate"])
            if df.empty:
                continue

            cycles = sorted(df["cycle no"].dropna().unique())
            cell_name = adr_file.stem.replace("_adr_data", "")

            fig, axes = plt.subplots(len(cycles), len(step_order), figsize=(12, 4 * len(cycles)), sharex=False)
            axes = np.atleast_2d(axes)

            for row, cycle in enumerate(cycles):
                cycle_df = df[df["cycle no"] == cycle]
                if cycle_df.empty:
                    continue

                for col, step in enumerate(step_order):
                    ax = axes[row, col]
                    bg = "#ffe8e8" if step.strip().upper() == "CC_DCHG" else "#e8ffe8"
                    ax.set_facecolor(bg)
                    step_df = cycle_df[cycle_df["step name"].str.strip() == step]
                    if step_df.empty:
                        ax.text(0.5, 0.5, f"No data for {step}", ha="center", va="center")
                        ax.axis("off")
                        continue
                    rates = sorted(step_df["c_rate"].dropna().unique())
                    cmap = get_cmap("viridis", max(len(rates), 1))
                    color_map = {rate: cmap(idx) for idx, rate in enumerate(rates)}

                    handles, labels = [], []
                    for step_no, grp in step_df.groupby("step no"):
                        c_rate = grp["c_rate"].iloc[0]
                        color = color_map.get(c_rate, "tab:gray")

                        window = voltage_windows.get(
                            step, (grp["volt(v)"].min(), grp["volt(v)"].max())
                        )
                        step_dv = dv[step] if isinstance(dv, dict) and step in dv else dv
                        resampled = resample_trace(grp, window, step_dv)
                        if resampled is None:
                            continue
                        volt, cap, soc = resampled
                        dq = np.gradient(cap, volt, edge_order=2)
                        if use_savgol_filter and savgol_window and savgol_window >= 3:
                            effective_window = savgol_window
                            if effective_window >= len(dq):
                                effective_window = len(dq) - 1 if len(dq) % 2 == 0 else len(dq)
                            if effective_window >= 3:
                                dq_dv = savgol_filter(dq, effective_window, savgol_poly)
                            else:
                                dq_dv = dq
                        else:
                            dq_dv = dq
                        valid = ~np.isnan(dq_dv)
                        if not valid.any():
                            continue

                        label = f"{abs(c_rate):.2f}C" if pd.notna(c_rate) else "Unknown C-rate"
                        y_vals = np.abs(dq_dv[valid]) / 1000.0
                        h = ax.plot(volt[valid], y_vals, color=color, linewidth=1.2)[0]
                        handles.append(h)
                        labels.append(label)

                        valid_volt = volt[valid]
                        valid_soc = soc[valid]
                        abs_vals = np.abs(dq_dv[valid])
                        peak_count = peaks_per_step.get(step, 3) if peaks_per_step else len(abs_vals)
                        peak_idx, _ = find_peaks(abs_vals)
                        low_priority_idx = None
                        if step.strip().upper() == "CCCV_CHG" and peak_idx.size:
                            low_candidates = [idx for idx in peak_idx if valid_volt[idx] < 3.32]
                            if low_candidates:
                                low_priority_idx = max(low_candidates, key=lambda idx: abs_vals[idx])
                        if peak_idx.size:
                            order = np.argsort(abs_vals[peak_idx])[::-1]
                            selected = peak_idx[order[:peak_count]].tolist()
                        else:
                            selected = []
                        if low_priority_idx is not None and low_priority_idx not in selected:
                            selected.append(low_priority_idx)
                        peak_idx = np.array(selected, dtype=int)
                        # Enforce CCCV peak rule: if the strongest peak < 3.32 V exists,
                        # label it as P3 regardless of its magnitude rank.
                        labels_for_idx: Dict[int, int] = {}
                        if peak_idx.size:
                            ordered_idx = peak_idx[np.argsort(abs_vals[peak_idx])[::-1]]
                            reserved_number = None
                            if step.strip().upper() == "CCCV_CHG":
                                low_candidates = [
                                    idx for idx in ordered_idx if valid_volt[idx] < 3.32
                                ]
                                if low_candidates:
                                    reserved_idx = low_candidates[0]
                                    labels_for_idx[reserved_idx] = 3
                                    reserved_number = 3
                            next_label = 1
                            for idx in ordered_idx:
                                if idx in labels_for_idx:
                                    continue
                                while reserved_number is not None and next_label == reserved_number:
                                    next_label += 1
                                labels_for_idx[idx] = next_label
                                next_label += 1
                        for idx in peak_idx:
                            peak_number = labels_for_idx.get(idx, 1)
                            y_val = abs_vals[idx] / 1000.0
                            ax.scatter(
                                valid_volt[idx],
                                y_val,
                                color=color,
                                s=10,
                                marker="x",
                            )
                            ax.text(
                                valid_volt[idx],
                                y_val,
                                f"P{peak_number}",
                                fontsize=6,
                                color="black",
                                ha="left",
                                va="bottom",
                            )
                            peaks_records.append(
                                {
                                    "cell_id": cell_name,
                                    "cycle no": cycle,
                                    "step no": step_no,
                                    "step name": step,
                                    "c_rate": c_rate,
                                    "peak_dq_dv": float(abs_vals[idx]),
                                    "voltage_at_peak": float(valid_volt[idx]),
                                    "soh_at_peak": float(valid_soc[idx]),
                                    "peak_label": f"P{peak_number}",
                                }
                            )

                    ax.set_ylabel("|dQ/dV| (kAh/V)")
                    ax.set_title(f"{cell_name} – Cycle {cycle} – {step}")
                    ax.grid(alpha=0.3)
                    ax.xaxis.set_major_locator(MultipleLocator(0.01))
                    ax.tick_params(axis="x", labelrotation=90)
                    if handles:
                        ax.legend(
                            handles,
                            labels,
                            fontsize=8,
                            loc="upper left",
                            ncol=1,
                            frameon=False,
                        )

            axes[-1, :].flat[-1].set_xlabel("Voltage (V)")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, orientation="landscape")
            plt.close(fig)

    peaks_df = pd.DataFrame(peaks_records)
    peaks_df.to_csv(peaks_csv, index=False)
    return output_pdf, peaks_csv


def plot_peak_trends(
    peaks_csv: PathLike = DEFAULT_PEAKS_CSV,
    output_pdf: PathLike = DEFAULT_PEAKS_TRENDS_PDF,
    step_order: Sequence[str] = DEFAULT_STEP_ORDER,
) -> Path:
    """Plot cycle-wise peak magnitudes, voltages, and SoH per cell."""

    peaks_csv = Path(peaks_csv)
    output_pdf = Path(output_pdf)
    if not peaks_csv.exists():
        raise FileNotFoundError(f"{peaks_csv} does not exist")

    df = pd.read_csv(peaks_csv)
    if df.empty:
        raise ValueError(f"{peaks_csv} is empty")

    numeric_cols = ["cycle no", "peak_dq_dv", "voltage_at_peak", "soh_at_peak", "c_rate"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cycle no", "peak_dq_dv", "voltage_at_peak", "soh_at_peak"])
    if df.empty:
        raise ValueError("No valid peak rows found for plotting.")

    if "peak_label" not in df.columns:
        df = df.sort_values(["cell_id", "cycle no", "step name", "c_rate", "voltage_at_peak"])
        df["peak_label"] = (
            df.groupby(["cell_id", "cycle no", "step name", "c_rate"]).cumcount() + 1
        ).map(lambda val: f"P{int(val)}")
    df["peak_rank"] = (
        df["peak_label"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
        .fillna(1)
        .astype(int)
    )

    df["c_rate"] = df["c_rate"].apply(snap_c_rate)
    df = df[df["step name"].str.strip().isin(step_order)].copy()
    if df.empty:
        raise ValueError("No rows matched requested step names.")

    cells = sorted(df["cell_id"].dropna().unique())
    if not cells:
        raise ValueError("No cell identifiers found in peak summary.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    symbol_caption = "▲ |dQ/dV| (primary), ○ SoH (secondary), ✕ Peak Voltage (tertiary)"

    ordered_steps: List[str] = []
    for name in ("CCCV_Chg", "CC_DChg"):
        if name in step_order:
            ordered_steps.append(name)
    ordered_steps.extend(step for step in step_order if step not in ordered_steps)
    ordered_steps = list(dict.fromkeys(ordered_steps))

    cycles = sorted(df["cycle no"].dropna().unique())
    max_peaks = int(df["peak_rank"].max()) if df["peak_rank"].notna().any() else 1
    cell_ids = sorted(df["cell_id"].dropna().unique())
    rng = np.random.default_rng(42)
    palette = rng.random((len(cell_ids), 3))

    with PdfPages(output_pdf) as pdf:
        for step in ordered_steps:
            step_df = df[df["step name"].str.strip() == step]
            if step_df.empty:
                continue
            for peak_rank in range(1, max_peaks + 1):
                peak_df = step_df[step_df["peak_rank"] == peak_rank]
                if peak_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(10, 4))
                ax2 = ax.twinx()
                ax3 = ax.twinx()
                ax3.spines["right"].set_position(("axes", 1.1))
                ax3.spines["right"].set_visible(True)
                bg = "#e8ffe8" if step.strip().upper() == "CCCV_CHG" else "#ffe8e8"
                ax.set_facecolor(bg)

                handles = []
                for cell in cell_ids:
                    cell_subset = peak_df[peak_df["cell_id"] == cell]
                    if cell_subset.empty:
                        continue
                    cell_subset = cell_subset.sort_values("cycle no")
                    x = cell_subset["cycle no"].to_numpy()
                    color = palette[cell_ids.index(cell)]
                    ax.scatter(
                        x,
                        cell_subset["peak_dq_dv"].abs() / 1000.0,
                        color=color,
                        marker="^",
                        s=25,
                        label=cell,
                    )
                    ax2.scatter(
                        x,
                        cell_subset["voltage_at_peak"],
                        color=color,
                        marker="x",
                        s=18,
                        alpha=0.85,
                    )
                    ax3.scatter(
                        x,
                        cell_subset["soh_at_peak"],
                        facecolors="none",
                        edgecolors=color,
                        marker="o",
                        s=20,
                        linewidths=0.7,
                    )
                    handles.append(
                        Line2D([], [], color=color, marker="^", linestyle="none", label=cell)
                    )

                if not handles:
                    ax.text(0.5, 0.5, "No data for these peaks", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks(cycles)
                ax.set_xticklabels([f"{int(c)}" for c in cycles])
                ax.set_ylabel("|dQ/dV| (kAh/V)")
                ax2.set_ylabel("Peak Voltage (V)")
                ax3.set_ylabel("SoH at Peak (%)")
                ax.set_xlabel("Cycle no")
                ax.grid(alpha=0.3, linewidth=0.5)
                ax.tick_params(axis="x", labelsize=9)
                ax.set_title(f"{step} – P{peak_rank}", fontsize=13)

                short_labels = [h.get_label().replace("RD_RateCapability_", "") for h in handles]
                cell_legend = fig.legend(
                    handles,
                    short_labels,
                    loc="upper center",
                    ncol=4,
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.98),
                )
                cell_legend.set_title("Cell ID", prop={"size": 9})

                fig.text(0.5, 0.06, symbol_caption, ha="center", fontsize=9)
                fig.tight_layout(rect=[0.04, 0.18, 0.98, 0.94])
                pdf.savefig(fig)
                plt.close(fig)

    return output_pdf


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ICA dQ/dV plots and peak summaries.")
    parser.add_argument("--adr-dir", type=Path, default=DEFAULT_ADR_DIR)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--peaks-csv", type=Path, default=DEFAULT_PEAKS_CSV)
    parser.add_argument("--dv", type=float, default=0.002)
    parser.add_argument("--dv-cc_dchg", type=float, help="Override ΔV for CC_DChg")
    parser.add_argument("--dv-cccv_chg", type=float, help="Override ΔV for CCCV_Chg")
    parser.add_argument("--savgol-window", type=int, default=None)
    parser.add_argument("--savgol-poly", type=int, default=3)
    parser.add_argument("--use-savgol-filter", action="store_true")
    parser.add_argument("--peaks-cc_dchg", type=int, default=3)
    parser.add_argument("--peaks-cccv_chg", type=int, default=3)
    parser.add_argument(
        "--window",
        nargs=4,
        type=float,
        metavar=("dchg_min", "dchg_max", "cccv_min", "cccv_max"),
        help="Override voltage windows for CC_DChg and CCCV_Chg.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    voltage_windows = None
    if args.window:
        voltage_windows = {
            "CC_DChg": (args.window[0], args.window[1]),
            "CCCV_Chg": (args.window[2], args.window[3]),
        }
    dv_map = args.dv
    if args.dv_cc_dchg or args.dv_cccv_chg:
        dv_map = {
            "CC_DChg": args.dv_cc_dchg if args.dv_cc_dchg else args.dv,
            "CCCV_Chg": args.dv_cccv_chg if args.dv_cccv_chg else args.dv,
        }
    pdf_path, peaks_path = generate_ica_plots(
        adr_dir=args.adr_dir,
        output_pdf=args.output_pdf,
        peaks_csv=args.peaks_csv,
        dv=dv_map,
        voltage_windows=voltage_windows,
        savgol_window=args.savgol_window,
        savgol_poly=args.savgol_poly,
        use_savgol_filter=args.use_savgol_filter,
        peaks_per_step={
            "CC_DChg": args.peaks_cc_dchg,
            "CCCV_Chg": args.peaks_cccv_chg,
        },
    )
    print(f"Saved ICA plots to {pdf_path}")
    print(f"Saved peak summary to {peaks_path}")


if __name__ == "__main__":
    main()
