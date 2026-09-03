"""Shared utilities for M_QTC artifact table printers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def load_json(path: str | Path) -> dict:
    """Load a JSON file and return the parsed dict."""
    return json.loads(Path(path).read_text())


def bold_min(values: Sequence[float], fmt: str = ".2f") -> list[str]:
    """Format values, wrapping the minimum in \\textbf{}."""
    best = min(values)
    formatted = [f"{v:{fmt}}" for v in values]
    all_tied = len(set(formatted)) == 1
    out: list[str] = []
    for v in values:
        s = f"{v:{fmt}}"
        if v == best and not all_tied:
            s = rf"\textbf{{{s}}}"
        out.append(s)
    return out


def bold_max(values: Sequence[float], fmt: str = ".3f") -> list[str]:
    """Format values, wrapping the maximum in \\textbf{}."""
    best = max(values)
    formatted = [f"{v:{fmt}}" for v in values]
    all_tied = len(set(formatted)) == 1
    out: list[str] = []
    for v in values:
        s = f"{v:{fmt}}"
        if v == best and not all_tied:
            s = rf"\textbf{{{s}}}"
        out.append(s)
    return out


def print_latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    *,
    col_spec: str | None = None,
    note: str = "",
    small: bool = True,
    tabcolsep: str | None = None,
    midrule_after: Sequence[int] = (),
) -> None:
    """Print a formatted LaTeX table body to stdout."""
    ncols = len(headers)
    if col_spec is None:
        col_spec = "l" + "r" * (ncols - 1)

    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(f"  \\caption{{{caption}}}")
    print(f"  \\label{{{label}}}")
    if small:
        print(r"  \small")
    if tabcolsep:
        print(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    print(f"  \\begin{{tabular}}{{{col_spec}}}")
    print(r"    \toprule")
    print("    " + " & ".join(headers) + r" \\")
    print(r"    \midrule")

    midrule_set = set(midrule_after)
    for idx, row in enumerate(rows):
        print("    " + " & ".join(row) + r" \\")
        if idx in midrule_set:
            print(r"    \midrule")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    if note:
        print(f"% {note}")
