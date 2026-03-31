from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("gold_breakout_intrabar_recent.ipynb")


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


cells = [
    markdown_cell(
        """# GC Recent Intrabar Diagnostic

This notebook is a recent-data diagnostic workflow for manual out-of-sample intrabar inspection. It is intentionally separate from the main historical backtest.

Scope limits:

- Data source is `yfinance` (`GC=F`), not the main historical futures dataset.
- Coverage is only the recent 7-day intraday window allowed by the source.
- This notebook is for data inspection, session reconstruction, and hourly context building.
- It is not a replacement for the main backtest and does not make long-history claims.
"""
    ),
    markdown_cell(
        """## 0. Configuration

The defaults are intentionally conservative: use a local cache first and refresh only when you explicitly set `force_refresh = True`.
"""
    ),
    code_cell(
        """from pathlib import Path

from gold_breakout_intrabar_recent import (
    RecentIntrabarConfig,
    recent_intrabar_config_frame,
)

config = RecentIntrabarConfig(
    ticker="GC=F",
    period="7d",
    interval="1m",
    cache_dir=Path("cache") / "yfinance_intrabar_recent",
    session_gap_threshold_minutes=30,
)

recent_intrabar_config_frame(config)"""
    ),
    markdown_cell(
        """## 1. Imports

The helper module handles cache paths, `yfinance` normalization, session construction, and hourly resampling.
"""
    ),
    code_cell(
        """import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from gold_breakout_intrabar_recent import (
    build_recent_intrabar_context,
    recent_intrabar_cache_paths,
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
plt.rcParams["figure.dpi"] = 120"""
    ),
    markdown_cell(
        """## 2. Load Cached Or Recent Data

Set `force_refresh = True` only when you want to download the latest 7-day sample from `yfinance`.
"""
    ),
    code_cell(
        """force_refresh = False

context = build_recent_intrabar_context(config, force_refresh=force_refresh)
minute_bars = context["minute_bars"]
session_table = context["session_table"]
hourly_bars = context["hourly_bars"]
metadata = context["metadata"]
cache_paths = context["cache_paths"]

display(pd.DataFrame([metadata]))
display(pd.DataFrame({"path_type": list(cache_paths.keys()), "path": [str(path) for path in cache_paths.values()]}))
print(f"minute rows: {len(minute_bars):,}")
print(f"sessions: {len(session_table):,}")
print(f"hourly rows: {len(hourly_bars):,}")"""
    ),
    markdown_cell(
        """## 3. Raw Data Preview

This view is just a sanity check on timestamp coverage, prices, and volume.
"""
    ),
    code_cell(
        """display(minute_bars.head(10))
display(minute_bars.tail(10))"""
    ),
    markdown_cell(
        """## 4. Session Reconstruction

Sessions are inferred from gaps larger than the configured threshold. This is a diagnostic heuristic for recent intraday data, not the canonical historical session engine.
"""
    ),
    code_cell(
        """display(session_table)"""
    ),
    markdown_cell(
        """## 5. Hourly Aggregation From 1m Bars

The hourly table is a convenience view that helps compare recent minute data against the main strategy's bar-based structure.
"""
    ),
    code_cell(
        """display(hourly_bars.head(24))
display(hourly_bars.tail(24))"""
    ),
    markdown_cell(
        """## 6. Static Recent Overview

This plot is deliberately simple: recent 1-minute closes, session boundaries, and hourly closes. Animation and richer intrabar logic belong in the next diagnostic step.
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(minute_bars["ts_event"], minute_bars["close"], color="steelblue", linewidth=0.8)
for session_open in session_table["session_open_ts"]:
    axes[0].axvline(session_open, color="grey", alpha=0.15, linewidth=0.8)
axes[0].set_title("Recent GC=F 1-Minute Close")
axes[0].set_ylabel("Price")
axes[0].grid(True, alpha=0.2)

axes[1].plot(hourly_bars["ts_event"], hourly_bars["close"], color="darkorange", linewidth=1.2, marker="o", markersize=3)
for session_open in session_table["session_open_ts"]:
    axes[1].axvline(session_open, color="grey", alpha=0.15, linewidth=0.8)
axes[1].set_title("Hourly Close Reconstructed From 1-Minute Bars")
axes[1].set_ylabel("Price")
axes[1].grid(True, alpha=0.2)

fig.tight_layout()
display(fig)
plt.close(fig)"""
    ),
    markdown_cell(
        """## 7. Next-Step Hooks

This notebook now provides:

- recent 1-minute data with local cache
- inferred session boundaries
- hourly aggregation
- cache metadata and file locations

The next intrabar commit can build on this foundation to add richer replay/animation and manual execution-logic inspection.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
