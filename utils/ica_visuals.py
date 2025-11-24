"""Advanced ICA visualizations for cell-wise degradation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PathLike = str | Path
ALLOWED_RATES = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


def snap_c_rate(val: float) -> float:
    if pd.isna(val):
        return np.nan
    sign = np.sign(val) if val else 1.0
    target = ALLOWED_RATES[np.argmin(np.abs(ALLOWED_RATES - abs(val)))]
    return sign * target


def _load_peak_data(peaks_csv: PathLike) -> pd.DataFrame:
    df = pd.read_csv(peaks_csv)
    numeric_cols = ["cycle no", "peak_dq_dv", "voltage_at_peak", "soh_at_peak", "c_rate"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cell_id", "cycle no", "peak_dq_dv", "voltage_at_peak", "soh_at_peak"])
    if df.empty:
        raise ValueError("No valid rows in peak summary.")
    if "peak_label" not in df.columns:
        df = df.sort_values(["cell_id", "cycle no", "step name", "c_rate", "voltage_at_peak"])
        df["peak_label"] = (
            df.groupby(["cell_id", "cycle no", "step name", "c_rate"]).cumcount() + 1
        ).map(lambda i: f"P{i}")
    df["peak_rank"] = (
        df["peak_label"].astype(str).str.extract(r"(\d+)").astype(float).fillna(1).astype(int)
    )
    df["c_rate"] = df["c_rate"].apply(snap_c_rate)
    return df


def plot_cell_trajectory_grid(
    peaks_csv: PathLike,
    output_pdf: PathLike,
) -> Path:
    """Per-cell grid: heatmap(|dQ/dV|), peak voltage lines, SoH lines."""

    df = _load_peak_data(peaks_csv)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    peak_order = sorted(df["peak_label"].unique(), key=lambda s: (len(s), s))
    cmap = sns.color_palette("viridis", n_colors=len(peak_order))
    color_map = {label: cmap[idx] for idx, label in enumerate(peak_order)}

    with PdfPages(output_pdf) as pdf:
        for cell, cell_df in df.groupby("cell_id"):
            cell_df = cell_df.sort_values(["cycle no", "peak_rank"])
            cycles = sorted(cell_df["cycle no"].unique())
            peak_labels = sorted(cell_df["peak_label"].unique(), key=lambda s: (len(s), s))
            grid = (
                cell_df.pivot_table(
                    index="peak_label",
                    columns="cycle no",
                    values="peak_dq_dv",
                    aggfunc="mean",
                )
                .reindex(index=peak_labels, columns=cycles)
                .astype(float)
            )
            fig = plt.figure(figsize=(10, 7))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
            ax_heat = fig.add_subplot(gs[0])
            ax_volt = fig.add_subplot(gs[1], sharex=ax_heat)
            ax_soh = fig.add_subplot(gs[2], sharex=ax_heat)

            sns.heatmap(
                grid.abs() / 1000.0,
                cmap="magma",
                cbar_kws={"label": "|dQ/dV| (kAh/V)"},
                ax=ax_heat,
                xticklabels=[int(c) for c in cycles],
                yticklabels=peak_labels,
                linewidths=0.3,
                linecolor="0.8",
            )
            ax_heat.set_title(f"{cell} – Peak Intensity Map")
            ax_heat.set_xlabel("")
            ax_heat.set_ylabel("Peak label")

            for label in peak_labels:
                subset = cell_df[cell_df["peak_label"] == label].sort_values("cycle no")
                if subset.empty:
                    continue
                color = color_map[label]
                ax_volt.plot(
                    subset["cycle no"],
                    subset["voltage_at_peak"],
                    label=label,
                    color=color,
                    marker="o",
                )
                ax_soh.plot(
                    subset["cycle no"],
                    subset["soh_at_peak"],
                    label=label,
                    color=color,
                    marker="o",
                )

            ax_volt.set_ylabel("Voltage at peak (V)")
            ax_soh.set_ylabel("SoH at peak (%)")
            ax_soh.set_xlabel("Cycle no")
            ax_volt.legend(
                ncol=min(4, len(peak_labels)),
                fontsize=8,
                loc="upper left",
                frameon=False,
            )
            ax_volt.grid(alpha=0.3)
            ax_soh.grid(alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf


def plot_cohort_radar(
    peaks_csv: PathLike,
    output_pdf: PathLike,
    metric_weights: Dict[str, float] | None = None,
) -> Path:
    """Radar chart per cycle capturing deviations from median per peak."""

    df = _load_peak_data(peaks_csv)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["peak_dq_dv", "voltage_at_peak", "soh_at_peak"]
    metric_weights = metric_weights or {"peak_dq_dv": 1.0, "voltage_at_peak": 1.0, "soh_at_peak": 1.0}

    with PdfPages(output_pdf) as pdf:
        for cycle, cyc_df in df.groupby("cycle no"):
            peaks = sorted(cyc_df["peak_label"].unique(), key=lambda s: (len(s), s))
            if not peaks:
                continue
            cols = min(3, len(peaks))
            rows = int(np.ceil(len(peaks) / cols))
            fig, axes = plt.subplots(
                rows,
                cols,
                subplot_kw={"projection": "polar"},
                figsize=(4 * cols, 4 * rows),
            )
            axes = np.atleast_2d(axes)
            for idx, peak in enumerate(peaks):
                ax = axes[idx // cols, idx % cols]
                peak_df = cyc_df[cyc_df["peak_label"] == peak]
                if peak_df.empty:
                    ax.set_axis_off()
                    continue
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                angles = np.concatenate([angles, angles[:1]])
                median_vals = peak_df[metrics].median()
                std_vals = peak_df[metrics].std(ddof=0).replace(0, np.nan)
                for cell, cell_df in peak_df.groupby("cell_id"):
                    normalized = (cell_df[metrics].iloc[0] - median_vals) / std_vals
                    normalized = normalized.fillna(0)
                    deltas = normalized.to_numpy()
                    weighted = deltas * np.array([metric_weights[m] for m in metrics])
                    values = np.concatenate([weighted, weighted[:1]])
                    ax.plot(angles, values, label=cell.split("_")[-1])
                ax.set_title(f"Cycle {int(cycle)} – {peak}")
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics, fontsize=8)
            for ax in axes.ravel()[len(peaks) :]:
                ax.set_axis_off()
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.02, 0.5),
                    loc="center left",
                    frameon=False,
                    title="Cell",
                )
                for ax in axes.ravel():
                    legend_obj = getattr(ax, "legend_", None)
                    if legend_obj:
                        legend_obj.remove()
            fig.tight_layout(rect=[0.02, 0.02, 0.82, 0.98])
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf


def plot_peak_health_scorecard(
    peaks_csv: PathLike,
    output_pdf: PathLike,
) -> Path:
    """Heatmap of composite peak health per cell and cycle."""

    df = _load_peak_data(peaks_csv)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    zcols = {}
    for col in ["peak_dq_dv", "voltage_at_peak", "soh_at_peak"]:
        z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        zcols[col] = z
    df["health_score"] = zcols["peak_dq_dv"] + zcols["voltage_at_peak"] + zcols["soh_at_peak"]

    with PdfPages(output_pdf) as pdf:
        for peak, peak_df in df.groupby("peak_label"):
            if peak_df.empty:
                continue
            pivot = (
                peak_df.pivot_table(
                    index="cell_id",
                    columns="cycle no",
                    values="health_score",
                    aggfunc="mean",
                )
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(10, max(4, pivot.shape[0] * 0.3)))
            sns.heatmap(
                pivot,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".1f",
                linewidths=0.3,
                linecolor="0.8",
                cbar_kws={"label": "Composite health score (z)"},
                ax=ax,
            )
            ax.set_title(f"Peak health scorecard – {peak}")
            ax.set_xlabel("Cycle no")
            ax.set_ylabel("Cell")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf


def plot_interactive_scatter(
    peaks_csv: PathLike,
    output_html: PathLike,
    step_filter: Sequence[str] | None = None,
) -> Path:
    """Interactive 3D scatter: cycle vs |dQ/dV| vs voltage, color by SoH."""

    df = _load_peak_data(peaks_csv)
    if step_filter:
        df = df[df["step name"].isin(step_filter)]
    df["abs_dqdv"] = df["peak_dq_dv"].abs()
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    fig = px.scatter_3d(
        df,
        x="cycle no",
        y="abs_dqdv",
        z="voltage_at_peak",
        color="soh_at_peak",
        size="abs_dqdv",
        symbol="peak_label",
        hover_data=["cell_id", "step name", "c_rate"],
        color_continuous_scale="Viridis",
        title="ICA degradation landscape",
    )
    fig.update_layout(scene=dict(xaxis_title="Cycle", yaxis_title="|dQ/dV| (A/V)", zaxis_title="Peak voltage (V)"))
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return output_html


def plot_peak_sankey(
    peaks_csv: PathLike,
    output_html: PathLike,
    step_name: str = "CCCV_Chg",
) -> Path:
    """Sankey diagram showing flows in |dQ/dV| between cycles for a given step."""

    df = _load_peak_data(peaks_csv)
    df = df[df["step name"].str.strip() == step_name]
    if df.empty:
        raise ValueError(f"No rows for step {step_name}")

    nodes: List[str] = []
    node_index: Dict[str, int] = {}
    links: Dict[Tuple[str, str], float] = {}

    for cell, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle no")
        prev_state = None
        for _, row in cell_df.iterrows():
            state = f"C{int(row['cycle no'])}-{row['peak_label']}"
            if state not in node_index:
                node_index[state] = len(nodes)
                nodes.append(state)
            if prev_state is not None:
                key = (prev_state, state)
                delta = abs(row["peak_dq_dv"])
                links[key] = links.get(key, 0.0) + delta
            prev_state = state

    if not links:
        raise ValueError("Not enough transitions for Sankey plot.")

    link_sources = [node_index[src] for src, _ in links.keys()]
    link_targets = [node_index[tgt] for _, tgt in links.keys()]
    link_values = list(links.values())

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=20, thickness=15, color="rgba(31,119,180,0.7)"),
            link=dict(source=link_sources, target=link_targets, value=link_values),
        )
    )
    fig.update_layout(title=f"Peak flow transitions – {step_name}")
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return output_html


def cluster_cells_by_peaks(
    peaks_csv: PathLike,
    output_csv: PathLike,
    n_clusters: int = 4,
    step_filter: Sequence[str] | None = None,
    aggfunc: str = "mean",
    random_state: int = 0,
) -> Path:
    """Cluster cells using aggregated peak metrics."""

    df = _load_peak_data(peaks_csv)
    if step_filter:
        df = df[df["step name"].str.strip().isin(step_filter)]
    if df.empty:
        raise ValueError("No rows available after filtering for clustering.")

    metrics = ["peak_dq_dv", "voltage_at_peak", "soh_at_peak"]
    feature_frames = []
    for metric in metrics:
        pivot = (
            df.pivot_table(
                index="cell_id",
                columns="peak_label",
                values=metric,
                aggfunc=aggfunc,
            )
            .sort_index(axis=1)
        )
        pivot = pivot.add_prefix(f"{metric}_")
        feature_frames.append(pivot)

    features = pd.concat(feature_frames, axis=1)
    if features.isnull().all(axis=None):
        raise ValueError("All feature values are NaN; cannot cluster.")

    features = features.fillna(features.mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)

    result = features.copy()
    result.insert(0, "cluster", labels)
    result.insert(1, "cell_id", result.index)
    result.reset_index(drop=True, inplace=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return output_csv


def evaluate_cluster_counts(
    peaks_csv: PathLike,
    output_csv: PathLike,
    k_values: Sequence[int] = (2, 3, 4, 5, 6),
    step_filter: Sequence[str] | None = None,
    aggfunc: str = "mean",
    random_state: int = 0,
) -> Path:
    """Evaluate silhouette scores across cluster counts."""

    df = _load_peak_data(peaks_csv)
    if step_filter:
        df = df[df["step name"].str.strip().isin(step_filter)]
    if df.empty:
        raise ValueError("No rows available after filtering for clustering.")

    metrics = ["peak_dq_dv", "voltage_at_peak", "soh_at_peak"]
    feature_frames = []
    for metric in metrics:
        pivot = (
            df.pivot_table(
                index="cell_id",
                columns="peak_label",
                values=metric,
                aggfunc=aggfunc,
            )
            .sort_index(axis=1)
        )
        pivot = pivot.add_prefix(f"{metric}_")
        feature_frames.append(pivot)

    features = pd.concat(feature_frames, axis=1)
    features = features.fillna(features.mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    rows = []
    for k in k_values:
        if k < 2:
            continue
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score, "inertia": kmeans.inertia_})

    summary = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return output_csv


def plot_cluster_projection(
    clusters_csv: PathLike,
    output_pdf: PathLike,
) -> Path:
    """Scatter plot of clustered cells in PCA space."""

    df = pd.read_csv(clusters_csv)
    if "cluster" not in df.columns or "cell_id" not in df.columns:
        raise ValueError("clusters_csv must contain 'cluster' and 'cell_id'.")
    feature_cols = [col for col in df.columns if col not in {"cluster", "cell_id"}]
    if not feature_cols:
        raise ValueError("No feature columns to project.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)

    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    palette = sns.color_palette("tab10", n_colors=len(df["cluster"].unique()))
    color_map = {cluster: palette[idx % len(palette)] for idx, cluster in enumerate(sorted(df["cluster"].unique()))}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster, grp in df.groupby("cluster"):
        ax.scatter(grp["pc1"], grp["pc2"], color=color_map[cluster], label=f"Cluster {cluster}", s=60, alpha=0.85)
        for _, row in grp.iterrows():
            ax.text(row["pc1"], row["pc2"], row["cell_id"].split("_")[-1], fontsize=7, ha="center", va="center")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cell clusters in PCA space")
    ax.axhline(0, color="0.7", linewidth=0.7)
    ax.axvline(0, color="0.7", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)
    return output_pdf


def plot_cluster_correlation_heatmap(
    clusters_csv: PathLike,
    output_pdf: PathLike,
    method: str = "pearson",
) -> Path:
    """Plot correlation heatmaps of cluster features."""

    df = pd.read_csv(clusters_csv)
    if "cluster" not in df.columns or "cell_id" not in df.columns:
        raise ValueError("clusters_csv must contain 'cluster' and 'cell_id'.")
    feature_cols = [col for col in df.columns if col not in {"cluster", "cell_id"}]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for cluster_id, grp in df.groupby("cluster"):
            if grp[feature_cols].shape[0] < 2:
                continue
            corr = grp[feature_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=0.3,
                linecolor="0.8",
                cbar_kws={"label": f"{method.title()} correlation"},
                ax=ax,
            )
            ax.set_title(f"Cluster {cluster_id} – feature correlations")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return output_pdf
