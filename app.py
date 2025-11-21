"""Interactive ICA dashboard for experimenting with single-cell dQ/dV plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter, find_peaks
from matplotlib.cm import get_cmap



ADR_DIR = Path("/home/kcv/Desktop/Rate_Capability/results/data/adr_data")
DEFAULT_WINDOWS: Dict[str, Tuple[float, float]] = {
    "CC_DChg": (2.85, 3.55),
    "CCCV_Chg": (2.70, 3.65),
}
ALLOWED_RATES = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


def snap_c_rate(val: float) -> float:
    if pd.isna(val):
        return np.nan
    sign = np.sign(val) if val else 1.0
    target = ALLOWED_RATES[np.argmin(np.abs(ALLOWED_RATES - abs(val)))]
    return sign * target


@st.cache_data(show_spinner=False)
def load_cell_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cycle no"] = pd.to_numeric(df["cycle no"], errors="coerce")
    df["c_rate_raw"] = pd.to_numeric(df["c_rate"], errors="coerce")
    skip_mask = np.isclose(df["c_rate_raw"].abs(), 0.33, atol=0.01)
    df = df[~skip_mask]
    df["c_rate"] = df["c_rate_raw"].apply(snap_c_rate)
    for col in ("volt(v)", "capacity(ah)", "soc(%)"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cycle no", "volt(v)", "capacity(ah)", "soc(%)", "c_rate"])
    return df


def resample_trace(
    grp: pd.DataFrame, window: Tuple[float, float], dv: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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


def compute_dq_dv(
    grp: pd.DataFrame,
    step: str,
    dv: float,
    window_overrides: Dict[str, Tuple[float, float]],
    savgol_window: int | None,
    savgol_poly: int,
):
    window = window_overrides.get(step, DEFAULT_WINDOWS.get(step, None))
    if window is None:
        window = (grp["volt(v)"].min(), grp["volt(v)"].max())

    resampled = resample_trace(grp, window, dv)
    if resampled is None:
        return None

    volt, cap, soc = resampled
    dq = np.gradient(cap, volt, edge_order=2)
    if savgol_window and savgol_window >= 3:
        effective_window = savgol_window
        if effective_window >= len(dq):
            effective_window = len(dq) - 1 if len(dq) % 2 == 0 else len(dq)
        if effective_window >= 3:
            dq = savgol_filter(dq, effective_window, savgol_poly)
    return volt, dq, soc, cap


def main() -> None:
    st.title("ICA Explorer")
    st.caption("Visualize dQ/dV curves per cell/cycle/step with adjustable parameters.")

    if not ADR_DIR.exists():
        st.error(f"ADR directory not found: {ADR_DIR}")
        return

    adr_files = sorted(ADR_DIR.glob("*_adr_data.csv"))
    if not adr_files:
        st.warning("No ADR datasets available.")
        return

    cell_options = {path.stem.replace("_adr_data", ""): path for path in adr_files}
    selected_cell = st.sidebar.selectbox("Cell", list(cell_options.keys()))
    selected_path = cell_options[selected_cell]

    df = load_cell_dataframe(selected_path)
    cycles = sorted(df["cycle no"].unique())
    cycle = st.sidebar.selectbox("Cycle", cycles)

    available_steps = sorted(df["step name"].str.strip().unique())
    steps = st.sidebar.multiselect("Steps", available_steps, default=list(DEFAULT_WINDOWS.keys()))

    manual_windows = {}
    with st.sidebar.expander("Voltage windows"):
        for step in steps:
            default = DEFAULT_WINDOWS.get(step, (float(df["volt(v)"].min()), float(df["volt(v)"].max())))
            vmin = st.number_input(f"{step} min V", value=float(default[0]), key=f"{step}-min")
            vmax = st.number_input(f"{step} max V", value=float(default[1]), key=f"{step}-max")
            manual_windows[step] = (vmin, vmax)

    cycle_df = df[df["cycle no"] == cycle]
    if cycle_df.empty:
        st.warning("No data for selected cycle.")
        return

    top_peaks = st.sidebar.number_input("Number of top peaks to show", min_value=1, value=1, step=1)

    def process_params(dv_value, sg_window_value, sg_poly_value, build_fig=True):
        fig_local = go.Figure() if build_fig else None
        cmap = get_cmap("viridis") if build_fig else None
        seen_rates_local = set()
        summary_records_local = []

        for step in steps:
            step_df = cycle_df[cycle_df["step name"].str.strip() == step]
            if step_df.empty:
                continue

            for step_no, grp in step_df.groupby("step no"):
                sg_window_used = (
                    int(sg_window_value) if sg_window_value and sg_window_value >= 3 else None
                )
                sg_poly_used = int(sg_poly_value) if sg_window_used else sg_poly_value
                result = compute_dq_dv(
                    grp,
                    step,
                    dv_value,
                    manual_windows,
                    sg_window_used,
                    sg_poly_used,
                )
                if result is None:
                    continue
                volt, dq_dv, soc, cap = result
                dq_abs = np.abs(dq_dv)
                dq_abs = np.abs(dq_dv)
                abs_rate = float(np.abs(grp["c_rate"].iloc[0]))
                rate_label = f"{abs_rate:.2f}C"

                if build_fig and fig_local is not None:
                    denom = np.ptp(ALLOWED_RATES) or 1.0
                    color = cmap((abs_rate - ALLOWED_RATES.min()) / denom)
                    fig_local.add_trace(
                        go.Scatter(
                            x=volt,
                            y=dq_abs,
                            mode="lines",
                            name=rate_label,
                            showlegend=rate_label not in seen_rates_local,
                            line=dict(
                                color=f"rgba({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f},1)",
                                width=1.6,
                            ),
                            hovertemplate="V=%{x:.3f} V<br>dQ/dV=%{y:.4f} Ah/V<br>SOC≈%{text:.1f}%<extra></extra>",
                            text=soc,
                        )
                    )
                    seen_rates_local.add(rate_label)

                valid = ~np.isnan(dq_abs)
                if valid.any():
                    capacity = cap[valid][-1] - cap[valid][0]
                    area = np.trapz(dq_abs[valid], volt[valid])
                    error_pct = abs(area - capacity) / abs(capacity) * 100.0 if capacity != 0 else np.nan
                    summary_records_local.append(
                        {
                            "C-rate": rate_label,
                            "Capacity between V window (Ah)": capacity,
                            "Area under ICA curve (Ah)": area,
                            "Error %": error_pct,
                        }
                    )
                    if build_fig and fig_local is not None:
                        peak_idx, _ = find_peaks(dq_abs[valid])
                        denom = np.ptp(ALLOWED_RATES) or 1.0
                        color = cmap((abs_rate - ALLOWED_RATES.min()) / denom)
                        if len(peak_idx) > 0:
                            sorted_idx = peak_idx[np.argsort(dq_abs[valid][peak_idx])[::-1][: int(top_peaks)]]
                            fig_local.add_trace(
                                go.Scatter(
                                    x=volt[valid][sorted_idx],
                                    y=dq_abs[valid][sorted_idx],
                                    mode="markers",
                                    showlegend=False,
                                    text=[f"Top {i+1}" for i in range(len(sorted_idx))],
                                    hovertemplate="Peak %{text}<br>V=%{x:.3f} V<br>dQ/dV=%{y:.4f}<extra></extra>",
                                    marker=dict(
                                        color=f"rgba({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f},0.9)",
                                        size=8,
                                        symbol="x",
                                    ),
                                )
                            )
                        else:
                            continue
                            fig_local.add_trace(
                                go.Scatter(
                                    x=[volt[valid][idx]],
                                    y=[dq_abs[valid][idx]],
                                    mode="markers",
                                    showlegend=False,
                                    text=[peak_number],
                                    hovertemplate="Peak %{text}<br>V=%{x:.3f} V<br>dQ/dV=%{y:.4f}<extra></extra>",
                                    marker=dict(
                                        color=f"rgba({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f},0.9)",
                                        size=7,
                                        symbol="x",
                                    ),
                                )
                            )

        summary_df_local = pd.DataFrame(summary_records_local)
        if build_fig and fig_local is not None:
            fig_local.update_layout(
                title=f"{selected_cell} – Cycle {cycle}",
                xaxis_title="Voltage (V)",
                yaxis_title="dQ/dV (Ah/V)",
                legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="right", x=1.02),
                template="plotly_white",
                height=700,
            )
        return fig_local, summary_df_local

    sg_window_default = None
    sg_poly_default = 3

    st.sidebar.subheader("Grid Search")
    dv_min = st.sidebar.number_input("ΔV min", value=0.001, min_value=0.0001, step=0.0005, format="%.4f")
    dv_max = st.sidebar.number_input("ΔV max", value=0.004, min_value=0.0001, step=0.0005, format="%.4f")
    dv_steps = st.sidebar.number_input("Number of ΔV steps", value=5, min_value=1, step=1)
    optimize_sg = st.sidebar.checkbox("Search Savitzky-Golay params", value=False)
    sg_windows_text = st.sidebar.text_input("SG windows (comma-separated)", "21,31,41")
    sg_poly_text = st.sidebar.text_input("SG polyorders", "2,3,4")

    best_result = st.session_state.get("ica_grid_best")
    if st.sidebar.button("Run grid search"):
        if dv_steps == 1:
            dv_candidates = np.array([dv_min])
        else:
            dv_candidates = np.linspace(dv_min, dv_max, dv_steps)
        sg_window_candidates = (
            [int(w.strip()) for w in sg_windows_text.split(",") if w.strip()]
            if optimize_sg
            else [sg_window_default]
        )
        sg_poly_candidates = (
            [int(p.strip()) for p in sg_poly_text.split(",") if p.strip()]
            if optimize_sg
            else [sg_poly_default]
        )
        best = None
        for dv_candidate in dv_candidates:
            for win_candidate in sg_window_candidates:
                for poly_candidate in sg_poly_candidates:
                    _, summary_df_candidate = process_params(
                        dv_candidate, win_candidate, poly_candidate, build_fig=False
                    )
                    if summary_df_candidate.empty:
                        continue
                    mean_error = summary_df_candidate["Error %"].dropna().mean()
                    if mean_error is None or np.isnan(mean_error):
                        continue
                    if best is None or mean_error < best["mean_error"]:
                        best = {
                            "dv": dv_candidate,
                            "sg_window": win_candidate,
                            "sg_poly": poly_candidate if win_candidate is not None else None,
                            "mean_error": float(mean_error),
                        }
        st.session_state["ica_grid_best"] = best
        best_result = best

    if best_result:
        st.sidebar.success(
            f"Best ΔV={best_result['dv']:.4f}, SG window={best_result['sg_window']}, "
            f"poly={best_result['sg_poly']} (mean error {best_result['mean_error']:.2f}%)"
        )
        if st.sidebar.button("Save best params to CSV"):
            save_dir = Path("/home/kcv/Desktop/Rate_Capability/results/data/ICA_best_params")
            save_dir.mkdir(parents=True, exist_ok=True)
            step_suffix = "_".join(sorted(set(step.strip() for step in steps))) or "all_steps"
            save_path = save_dir / f"ica_best_params_{selected_cell}_cycle{cycle}_{step_suffix}.csv"
            params_df = pd.DataFrame(
                [
                    {
                        "cell_id": selected_cell,
                        "cycle_no": cycle,
                        "steps": ",".join(steps),
                        "step_names": ";".join(sorted(set(step.strip() for step in steps))),
                        "dv": best_result["dv"],
                        "sg_window": best_result["sg_window"],
                        "sg_poly": best_result["sg_poly"],
                        "windows": manual_windows,
                    }
                ]
            )
            params_df.to_csv(save_path, index=False)
            summary_df_current = st.session_state.get("ica_last_summary")
            if summary_df_current is not None and not summary_df_current.empty:
                summary_path = save_dir / f"ica_capacity_vs_area_{selected_cell}_cycle{cycle}_{step_suffix}.csv"
                summary_df_current.to_csv(summary_path, index=False)
                st.sidebar.success(f"Saved params to {save_path}\nSaved summary to {summary_path}")
            else:
                st.sidebar.success(f"Saved best params to {save_path}")

    st.sidebar.markdown("---")
    use_best_params = st.sidebar.checkbox("Use grid-search best parameters if available", value=False)

    if use_best_params:
        if not best_result:
            st.sidebar.warning("Run grid search first to populate best parameters.")
            use_best_params = False
        else:
            dv = best_result["dv"]
            sg_window = best_result["sg_window"]
            sg_poly = best_result["sg_poly"]
            sg_text = f"SG window={sg_window}, poly={sg_poly}" if sg_window is not None else "No smoothing"
            st.sidebar.info(f"Using ΔV={dv:.4f}, {sg_text} from grid search.")
    if not use_best_params:
        dv = st.sidebar.number_input(
            "Voltage spacing (ΔV)", value=0.002, min_value=0.0005, step=0.0005, format="%.4f"
        )
        use_savgol = st.sidebar.checkbox("Apply Savitzky-Golay smoothing", value=True)
        if use_savgol:
            sg_window = st.sidebar.number_input("Savitzky-Golay window (odd)", min_value=3, step=2, value=51)
            sg_poly = st.sidebar.number_input("Savitzky-Golay polyorder", min_value=1, value=3, step=1)
        else:
            sg_window = None
            sg_poly = sg_poly_default

    fig, summary_df = process_params(dv, sg_window, sg_poly, build_fig=True)

    if not summary_df.empty:
        summary_display = summary_df[
            ["C-rate", "Capacity between V window (Ah)", "Area under ICA curve (Ah)", "Error %"]
        ]
        st.session_state["ica_last_summary"] = summary_display
        col_plot, col_table = st.columns([1.5, 1])
        with col_plot:
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.subheader("Capacity vs ICA Area")
            st.dataframe(summary_display, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
