"""Generate OCV-enhanced datasets by combining SoC, resistance, and C-rate info."""

from __future__ import annotations

import argparse
import pickle
import sys
from importlib import import_module
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .resistance_analysis import save_cell_resistance
from .soc_data import get_soc_dataframe

PathLike = Union[str, Path]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "Data"
DEFAULT_SOC_DIR = REPO_ROOT / "results" / "data" / "soc_data"
DEFAULT_RESISTANCE_DIR = REPO_ROOT / "results" / "data" / "resistance_data"
DEFAULT_C_RATE_STATS = REPO_ROOT / "results" / "data" / "c_rate_stats.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "data" / "ocv_power_data"


def _find_column(df: pd.DataFrame, desired: str) -> str | None:
    desired_lower = desired.lower()
    for col in df.columns:
        if str(col).lower() == desired_lower:
            return col
    return None


def max_capacities(
    df: pd.DataFrame,
    *,
    cycle_no: int | None = None,
    step_no: int | None = None,
    step_name: str | None = None,
) -> tuple[float, float]:
    """Return max charge/discharge capacities, optionally filtered by step metadata."""

    subset = df.copy()
    cycle_col = _find_column(subset, "cycle no")
    step_no_col = _find_column(subset, "step no")
    step_col = _find_column(subset, "step name")
    cap_col = _find_column(subset, "capacity(ah)")

    if cycle_no is not None and cycle_col is not None:
        subset = subset[subset[cycle_col] == cycle_no]
    if step_no is not None and step_no_col is not None:
        subset = subset[subset[step_no_col] == step_no]
    if step_name is not None and step_col is not None:
        cmp = str(step_name).strip().lower()
        subset = subset[
            subset[step_col].astype(str).str.strip().str.lower() == cmp
        ]

    if step_col is None or cap_col is None:
        raise ValueError("Required columns ('step name', 'capacity(ah)') not found.")
    if subset.empty:
        raise ValueError("No rows left after applying the requested filters.")

    chg_mask = subset[step_col].astype(str).str.contains("Chg", case=False, na=False)
    dch_mask = subset[step_col].astype(str).str.contains("DChg", case=False, na=False)
    qchg = pd.to_numeric(subset.loc[chg_mask, cap_col], errors="coerce").abs().max()
    qdchg = pd.to_numeric(subset.loc[dch_mask, cap_col], errors="coerce").abs().max()
    return float(qchg), float(qdchg)


class OCVEstimator:
    """Wraps cluster-based OCV inference."""

    def __init__(self, cluster_obj) -> None:
        self.cluster_obj = cluster_obj

    def calculate_ocv_from_soc(
        self,
        soc_df: pd.DataFrame,
        *,
        cycle_no: int | None = None,
        step_no: int | None = None,
        step_name: str | None = None,
    ) -> pd.DataFrame:
        """Attach OCV estimates derived from the learned mid-curves."""
        if self.cluster_obj is None:
            raise ValueError("cluster_obj is required to compute OCV from SOC.")

        qchg, qdchg = max_capacities(
            soc_df,
            cycle_no=cycle_no,
            step_no=step_no,
            step_name=step_name,
        )
        cluster_id = int(self.cluster_obj.predict_cluster_from_capacity(Qchg=qchg, Qdchg=qdchg))
        soc_df = soc_df.copy()
        soc_df["OCV"] = self.cluster_obj.voltage_from_soc(cluster_id, soc_df["SOC"].to_numpy(float))
        return soc_df


def _load_cluster_model(path: PathLike):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Cluster model not found: {model_path}")
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def _load_cluster_cells_from_repo(
    repo_root: PathLike,
    cache_dir: PathLike,
    bin_size: float,
    k_clusters: int,
    embed_dims: int,
    random_state: int = 0,
):
    """Import ClusterCells from an external repo and instantiate/fit it."""

    repo_root = Path(repo_root)
    cache_dir = Path(cache_dir)
    utils_dir = repo_root / "utils"
    if not utils_dir.exists():
        raise FileNotFoundError(f"Could not find utils directory at {utils_dir}")

    sys_path_added = False
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
        sys_path_added = True

    try:
        cluster_module = import_module("cluster_cells")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"Unable to import cluster_cells from {utils_dir}. "
            f"Ensure the REPT_RPT_OCV_analysis repository is available."
        ) from exc

    ClusterCells = getattr(cluster_module, "ClusterCells")
    cluster_obj = ClusterCells(
        CACHE_DIR=cache_dir,
        BIN=bin_size,
        K=k_clusters,
        EMBED_DIMS=embed_dims,
        random_state=random_state,
    )
    cluster_obj.fit()

    if sys_path_added:
        sys.path.remove(str(utils_dir))

    return cluster_obj


def _ensure_soc_dataframe(cell_csv: PathLike, soc_dir: PathLike) -> Tuple[pd.DataFrame, str]:
    df, cell_name = get_soc_dataframe(cell_csv, output_dir=soc_dir)
    df = df.reset_index(drop=True)
    df["row_index"] = np.arange(len(df), dtype=int)
    return df, cell_name


def _ensure_resistance_dataframe(cell_csv: PathLike, resistance_dir: PathLike) -> pd.DataFrame:
    resistance_dir = Path(resistance_dir)
    resistance_dir.mkdir(parents=True, exist_ok=True)
    resistance_path = resistance_dir / f"{Path(cell_csv).stem}_resistance.csv"
    if not resistance_path.exists():
        save_cell_resistance(cell_csv, resistance_dir)
    res_df = pd.read_csv(resistance_path)
    if "row_index" in res_df.columns:
        res_df["row_index"] = res_df["row_index"].astype(int)
    return res_df


def _load_c_rate_stats(path: PathLike) -> pd.DataFrame:
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"C-rate stats CSV not found: {stats_path}")
    return pd.read_csv(stats_path)


def build_cell_ocv_table(
    cell_csv: PathLike,
    estimator: OCVEstimator,
    stats_df: pd.DataFrame,
    soc_dir: PathLike = DEFAULT_SOC_DIR,
    resistance_dir: PathLike = DEFAULT_RESISTANCE_DIR,
    output_dir: PathLike = DEFAULT_OUTPUT_DIR,
) -> Tuple[pd.DataFrame, Path]:
    """Return the enriched dataset for a single cell and persist it."""

    soc_df, cell_name = _ensure_soc_dataframe(cell_csv, soc_dir)

    ocv_input = soc_df.rename(
        columns={
            "step name": "Step name",
            "capacity(ah)": "Capacity(Ah)",
            "soc(%)": "SOC",
        }
    )[["Step name", "Capacity(Ah)", "SOC"]]
    ocv_input["Step name"] = ocv_input["Step name"].astype(str)
    ocv_input["Capacity(Ah)"] = pd.to_numeric(ocv_input["Capacity(Ah)"], errors="coerce")
    ocv_input["SOC"] = pd.to_numeric(ocv_input["SOC"], errors="coerce")
    ocv_augmented = estimator.calculate_ocv_from_soc(ocv_input)
    soc_df["OCV"] = ocv_augmented["OCV"].to_numpy()

    res_df = _ensure_resistance_dataframe(cell_csv, resistance_dir)
    res_lookup = res_df.set_index("row_index")["resistance(ohm)"]
    soc_df["resistance(ohm)"] = soc_df["row_index"].map(res_lookup)

    cell_stats = stats_df[stats_df["cell_name"] == cell_name]
    if not cell_stats.empty:
        c_rate_lookup = (
            cell_stats.set_index(["cycle no", "step no"]).sort_index()["mean_c_rate"]
        )
        soc_df["c_rate"] = [
            c_rate_lookup.get((cycle, step), np.nan)
            for cycle, step in zip(soc_df["cycle no"], soc_df["step no"])
        ]
    else:
        soc_df["c_rate"] = np.nan

    volt_arr = pd.to_numeric(soc_df["volt(v)"], errors="coerce").to_numpy(float)
    res_arr = pd.to_numeric(soc_df["resistance(ohm)"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.where(np.abs(res_arr) > 1e-12, (volt_arr**2) / (4.0 * res_arr), np.nan)
    soc_df["power(w)"] = power

    out_cols = [
        "cell_name",
        "cycle no",
        "step no",
        "step name",
        "capacity(ah)",
        "c_rate",
        "current(a)",
        "volt(v)",
        "soc(%)",
        "resistance(ohm)",
        "OCV",
        "power(w)",
    ]
    out_df = soc_df.loc[:, out_cols].copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{cell_name}_ocv_data.csv"
    out_df.to_csv(out_path, index=False)
    return out_df, out_path


def process_all_cells(
    data_dir: PathLike,
    estimator: OCVEstimator,
    stats_df: pd.DataFrame,
    soc_dir: PathLike = DEFAULT_SOC_DIR,
    resistance_dir: PathLike = DEFAULT_RESISTANCE_DIR,
    output_dir: PathLike = DEFAULT_OUTPUT_DIR,
    glob: str = "RD_RateCapability_*.csv",
) -> List[Path]:
    data_path = Path(data_dir)
    cell_paths = sorted(data_path.glob(glob))
    if not cell_paths:
        raise FileNotFoundError(f"No CSV files matching '{glob}' under {data_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for cell_file in cell_paths:
        _, out_path = build_cell_ocv_table(
            cell_file,
            estimator=estimator,
            stats_df=stats_df,
            soc_dir=soc_dir,
            resistance_dir=resistance_dir,
            output_dir=output_dir,
        )
        written.append(out_path)
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCV-enhanced datasets per cell.")
    parser.add_argument("--cluster-model", type=Path, help="Pickled cluster object path.")
    parser.add_argument("--cluster-repo", type=Path, help="Path to REPT_RPT_OCV_analysis repository.")
    parser.add_argument("--cluster-cache", type=Path, help="Directory with soc_voltage_cache*.npz files.")
    parser.add_argument("--cluster-bin", type=float, help="BIN value used when training the cluster model.")
    parser.add_argument("--cluster-k", type=int, help="Number of clusters (K).")
    parser.add_argument("--cluster-embed-dims", type=int, default=2, help="Embedding dimensions for PCA.")
    parser.add_argument("--cluster-random-state", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--soc-dir", type=Path, default=DEFAULT_SOC_DIR)
    parser.add_argument("--resistance-dir", type=Path, default=DEFAULT_RESISTANCE_DIR)
    parser.add_argument("--c-rate-stats", type=Path, default=DEFAULT_C_RATE_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glob", default="RD_RateCapability_*.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.cluster_model:
        cluster_obj = _load_cluster_model(args.cluster_model)
    else:
        required = [args.cluster_repo, args.cluster_cache, args.cluster_bin, args.cluster_k]
        if not all(required):
            raise SystemExit(
                "Provide either --cluster-model (pickle) or "
                "--cluster-repo/--cluster-cache/--cluster-bin/--cluster-k values."
            )
        cluster_obj = _load_cluster_cells_from_repo(
            repo_root=args.cluster_repo,
            cache_dir=args.cluster_cache,
            bin_size=args.cluster_bin,
            k_clusters=args.cluster_k,
            embed_dims=args.cluster_embed_dims,
            random_state=args.cluster_random_state,
        )
    estimator = OCVEstimator(cluster_obj)
    stats_df = _load_c_rate_stats(args.c_rate_stats)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_dir)
    cell_paths = sorted(data_path.glob(args.glob))
    if not cell_paths:
        raise SystemExit(f"No cell CSV files found in {data_path} matching {args.glob}")

    written = []
    for cell_csv in cell_paths:
        out_df, out_path = build_cell_ocv_table(
            cell_csv,
            estimator=estimator,
            stats_df=stats_df,
            soc_dir=args.soc_dir,
            resistance_dir=args.resistance_dir,
            output_dir=args.output_dir,
        )
        written.append(out_path)
        print(f"[OK] {cell_csv.name} -> {out_path}")

    print(f"Wrote {len(written)} OCV datasets to {output_dir}")


if __name__ == "__main__":
    main()
