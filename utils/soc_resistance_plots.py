"""Utilities to plot SoC vs resistance mosaics per cell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOC_DIR = REPO_ROOT / "results" / "data" / "soc_data"
DEFAULT_RESISTANCE_DIR = REPO_ROOT / "results" / "data" / "resistance_data"
DEFAULT_C_RATE_STATS = REPO_ROOT / "results" / "data" / "c_rate_stats.csv"
DEFAULT_OUTPUT_PDF = REPO_ROOT / "results" / "plots" / "soc_vs_resistance.pdf"


def _normalize_step(step: object) -> str:
    return str(step).strip().lower()


def _load_soc_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["row_index"] = np.arange(len(df), dtype=int)
    df["step_norm"] = df["step name"].map(_normalize_step)
    return df


def _load_resistance_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_index" in df.columns:
        df["row_index"] = df["row_index"].astype(int)
    df["step_norm"] = df["step name"].map(_normalize_step)
    return df


def _load_c_rate_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["step_norm"] = df["step name"].map(_normalize_step)
    return df


def _build_c_rate_color_map(c_rates: pd.Series) -> Dict[float, str]:
    unique_rates = sorted({float(round(abs(rate), 6)) for rate in c_rates.dropna()})
    if not unique_rates:
        return {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rates)))
    return {rate: plt.matplotlib.colors.to_hex(color) for rate, color in zip(unique_rates, colors)}


def plot_cell_soc_resistance(
    cell_name: str,
    soc_df: pd.DataFrame,
    resistance_df: pd.DataFrame,
    c_rate_df: pd.DataFrame,
    pdf: PdfPages,
    steps: Sequence[Tuple[str, str]] = (("cccv_chg", "CCCV_Chg"), ("cc_dchg", "CC_DChg")),
) -> None:
    """Plot SoC vs resistance grids for a single cell (one cycle per PDF page)."""

    soc_df = soc_df.reset_index(drop=True)
    soc_indexed = soc_df.set_index("row_index")

    cycles = sorted(soc_df["cycle no"].dropna().unique())
    if not cycles:
        return

    cell_c_rates = c_rate_df[c_rate_df["cell_name"] == cell_name].copy()
    rate_color_map = _build_c_rate_color_map(cell_c_rates["mean_c_rate"])

    resistance_df = resistance_df.sort_values("row_index").reset_index(drop=True)

    for cycle_no in cycles:
        fig_height = 2.6 * len(steps) + 0.4
        fig = plt.figure(figsize=(6.4, fig_height))
        cycle_mask = soc_df["cycle no"] == cycle_no
        cycle_c_rates = cell_c_rates[cell_c_rates["cycle no"] == cycle_no]
        cycle_res = resistance_df[resistance_df["cycle no"] == cycle_no]

        title_top = 0.94
        avail_height = title_top - 0.08
        ax_height = avail_height / len(steps)
        current_bottom = title_top - ax_height

        for step_norm, step_label in steps:
            ax = fig.add_axes([0.12, current_bottom, 0.8, ax_height - 0.05])
            step_entries = cycle_c_rates[cycle_c_rates["step_norm"] == step_norm]
            plotted = False
            res_step = cycle_res[cycle_res["step_norm"] == step_norm]
            for _, step_row in step_entries.iterrows():
                step_no = int(step_row["step no"])
                c_rate = float(step_row.get("mean_c_rate", np.nan))
                mask = cycle_mask & (soc_df["step no"] == step_no)
                soc_indices = soc_df.loc[mask, "row_index"].to_numpy(dtype=int)
                res_subset = res_step[res_step["step no"] == step_no]
                if res_subset.empty or soc_indices.size == 0:
                    continue

                res_subset = res_subset.dropna(subset=["resistance(ohm)"])
                if res_subset.empty:
                    continue

                res_subset = res_subset[res_subset["row_index"].isin(soc_indices)]
                if res_subset.empty:
                    continue

                row_order = res_subset["row_index"].to_numpy(dtype=int)
                res_values = res_subset["resistance(ohm)"].to_numpy(dtype=float)
                order = np.argsort(row_order)
                row_order = row_order[order]
                res_values = res_values[order]
                soc_values = soc_indexed.loc[row_order, "soc(%)"].to_numpy(dtype=float)

                valid = (~np.isnan(res_values)) & (~np.isnan(soc_values))
                if not valid.any():
                    continue

                soc_values = soc_values[valid]
                res_values = res_values[valid] * 1000.0  # convert to mΩ

                rate_key = float(round(abs(c_rate), 6))
                color = rate_color_map.get(rate_key, "#1f77b4")
                decimals = 1 if step_norm == "cccv_chg" else 2
                label = f"{abs(c_rate):.{decimals}f}C"
                ax.plot(soc_values, res_values, color=color, linewidth=1.0, label=label)
                plotted = True

            ax.set_title(f"Cycle {cycle_no} – {step_label}")
            ax.set_ylabel("Resistance (mΩ)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            if step_norm == steps[-1][0]:
                ax.set_xlabel("SoC (%)")
            else:
                ax.set_xticklabels([])

            if plotted:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(
                        handles,
                        labels,
                        fontsize=7,
                        loc="upper center",
                        ncol=len(labels),
                        frameon=False,
                    )
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")

            current_bottom -= ax_height

        fig.suptitle(f"SoC vs Resistance – {cell_name}", fontsize=14, y=0.97)
        pdf.savefig(fig)
        plt.close(fig)


def generate_soc_resistance_pdf(
    soc_dir: Path,
    resistance_dir: Path,
    output_pdf: Path,
    c_rate_stats_path: Path,
    steps: Sequence[Tuple[str, str]] = (("cccv_chg", "CCCV_Chg"), ("cc_dchg", "CC_DChg")),
) -> List[Path]:
    """Generate a single multi-page PDF with one page per cell."""

    soc_paths = sorted(soc_dir.glob("*_soc_data.csv"))
    if not soc_paths:
        raise FileNotFoundError(f"No SoC data CSVs found in {soc_dir}")

    c_rate_stats = _load_c_rate_dataframe(c_rate_stats_path)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    with PdfPages(output_pdf) as pdf:
        for soc_path in soc_paths:
            cell_name = soc_path.stem.replace("_soc_data", "")
            resistance_path = resistance_dir / f"{cell_name}_resistance.csv"
            if not resistance_path.exists():
                print(f"[WARN] Missing resistance file for {cell_name}; skipping.")
                continue

            soc_df = _load_soc_dataframe(soc_path)
            resistance_df = _load_resistance_dataframe(resistance_path)
            plot_cell_soc_resistance(
                cell_name,
                soc_df,
                resistance_df,
                c_rate_stats,
                pdf,
                steps=steps,
            )
            written.append(resistance_path)

    return written


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SoC vs resistance grids and save to a PDF.")
    parser.add_argument("--soc-dir", type=Path, default=DEFAULT_SOC_DIR, help="Directory with *_soc_data.csv files.")
    parser.add_argument(
        "--resistance-dir",
        type=Path,
        default=DEFAULT_RESISTANCE_DIR,
        help="Directory containing *_resistance.csv tables.",
    )
    parser.add_argument(
        "--c-rate-stats",
        type=Path,
        default=DEFAULT_C_RATE_STATS,
        help="Path to c_rate_stats.csv (used for labeling C-rates).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Path to the resulting PDF file.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    written = generate_soc_resistance_pdf(
        args.soc_dir,
        args.resistance_dir,
        args.output,
        args.c_rate_stats,
    )
    print(f"Generated PDF at {args.output} for {len(written)} cells.")


if __name__ == "__main__":
    main()
