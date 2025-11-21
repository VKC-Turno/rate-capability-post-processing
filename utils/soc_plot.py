"""Plot helpers for SoC-vs-Voltage visualizations."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .soc_data import compute_soc_series


def _downsample(soc: np.ndarray, volt: np.ndarray, max_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    if soc.size <= max_points:
        return soc, volt
    idx = np.linspace(0, soc.size - 1, max_points).astype(int)
    return soc[idx], volt[idx]


def _segment_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if x.size < 3:
        return float("inf"), 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    sse = float(np.sum((y - pred) ** 2))
    return sse, slope, intercept


def bacon_watts_knees(
    soc: np.ndarray,
    volt: np.ndarray,
    step_name: Optional[str] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Approximate two knees via brute-force Bacon-Watts style fit."""

    if soc.size < 10:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))

    ds_soc, ds_volt = _downsample(soc, volt)
    n = ds_soc.size
    min_gap = max(3, n // 50)

    best_err = float("inf")
    best_breaks = (float("nan"), float("nan"))

    for i in range(min_gap, n - 2 * min_gap):
        for j in range(i + min_gap, n - min_gap):
            sse1, _, _ = _segment_fit(ds_soc[:i], ds_volt[:i])
            sse2, _, _ = _segment_fit(ds_soc[i:j], ds_volt[i:j])
            sse3, _, _ = _segment_fit(ds_soc[j:], ds_volt[j:])
            err = sse1 + sse2 + sse3
            if err < best_err:
                best_err = err
                best_breaks = (ds_soc[i], ds_soc[j])

    low_soc, high_soc = best_breaks
    if np.isnan(low_soc) or np.isnan(high_soc):
        return (float("nan"), float("nan")), (float("nan"), float("nan"))

    low_volt = float(np.interp(low_soc, soc, volt))
    high_volt = float(np.interp(high_soc, soc, volt))
    return (low_soc, low_volt), (high_soc, high_volt)


def windowed_knees(
    soc: np.ndarray,
    volt: np.ndarray,
    windows: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def _find_peak(bounds):
        soc_bounds, volt_bounds = bounds
        mask = (
            (soc >= soc_bounds[0])
            & (soc <= soc_bounds[1])
            & (volt >= volt_bounds[0])
            & (volt <= volt_bounds[1])
        )
        if mask.sum() < 3:
            return float("nan"), float("nan")

        s = soc[mask]
        v = volt[mask]
        order = np.argsort(s)
        s = s[order]
        v = v[order]

        dv = np.gradient(v, s, edge_order=2)
        d2v = np.gradient(dv, s, edge_order=2)
        idx = int(np.argmax(np.abs(d2v)))
        return float(s[idx]), float(v[idx])

    knees = [_find_peak(win) for win in windows]
    return knees[0], knees[1]


def knee_points(soc: np.ndarray, volt: np.ndarray, step_name: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return knees for the provided trace (with CC_DChg-specific windows)."""

    step_clean = step_name.strip().lower()
    if step_clean == "cc_dchg":
        low_win = ((80.0, 100.0), (2.95, 3.35))
        high_win = ((0.0, 30.0), (2.8, 3.2))
        knees = windowed_knees(soc, volt, (low_win, high_win))
        if not np.isnan(knees[0][0]) and not np.isnan(knees[1][0]):
            return knees
        # fallback to Bacon-Watts if the windows failed
    return bacon_watts_knees(soc, volt, step_name=step_name)


def prepare_soc_trace(
    subset: pd.DataFrame, step_name: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Return SoC (%) vs voltage arrays plus knee point info."""

    if "soc(%)" in subset.columns:
        soc = subset["soc(%)"].to_numpy()
    else:
        if "record number" in subset.columns:
            subset = subset.sort_values("record number").reset_index(drop=True)
        else:
            subset = subset.sort_index().reset_index(drop=True)
        soc = compute_soc_series(subset).values

    order = np.argsort(soc)
    soc = soc[order]
    volt = subset["volt(v)"].values[order]
    low_knee, high_knee = knee_points(soc, volt, step_name)
    return soc, volt, low_knee, high_knee


def plot_soc_curve(
    ax: plt.Axes,
    trace: pd.DataFrame,
    step_name: str,
    color: Optional[str],
    label: str,
    show_knees: bool = True,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Plot a SoC-V trace with optional knee markers."""

    soc, volt, low_knee, high_knee = prepare_soc_trace(trace, step_name)
    ax.plot(soc, volt, label=label, color=color)
    if show_knees:
        for knee_soc, knee_volt in (low_knee, high_knee):
            if not np.isnan(knee_soc):
                ax.scatter(
                    knee_soc,
                    knee_volt,
                    color=color,
                    marker="x",
                    s=30,
                    linewidths=1.2,
                )
    return low_knee, high_knee


def piecewise_knees(
    subset: pd.DataFrame,
    step_name: str,
    segments: int = 7,
    low_index: int = 2,
    high_index: int = 6,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Piecewise-linear knees using `segments` breakpoints."""

    if "soc(%)" not in subset.columns:
        subset = subset.copy()
        subset["soc(%)"] = compute_soc_series(subset)

    subset = subset.sort_values("soc(%)")
    soc = subset["soc(%)"].values
    volt = subset["volt(v)"].values

    try:
        from pwlf import PiecewiseLinFit
    except ImportError:
        return bacon_watts_knees(soc, volt, step_name=step_name)

    model = PiecewiseLinFit(soc, volt)
    breakpoints = model.fit(segments)
    inner_breaks = breakpoints[1:-1]
    if len(inner_breaks) < max(low_index, high_index):
        return bacon_watts_knees(soc, volt, step_name=step_name)

    low_soc = inner_breaks[low_index - 1]
    high_soc = inner_breaks[high_index - 1]
    low_volt = model.predict([low_soc])[0]
    high_volt = model.predict([high_soc])[0]
    return (low_soc, low_volt), (high_soc, high_volt)


__all__ = [
    "bacon_watts_knees",
    "knee_points",
    "prepare_soc_trace",
    "plot_soc_curve",
    "piecewise_knees",
]
