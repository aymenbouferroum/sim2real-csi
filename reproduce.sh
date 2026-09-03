#!/usr/bin/env bash
# Reproduce all paper results from the shipped data.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p figures_out

# ── Tables ────────────────────────────────────────────────────────
echo "── Tables ──"
python3 tables/print_table_II.py
python3 tables/print_table_III.py
python3 tables/print_table_IV.py
python3 tables/print_table_V.py

# ── Figures ───────────────────────────────────────────────────────
echo "── Figures ──"
for dir in Figures/Figure*/; do
    script=$(find "$dir" -maxdepth 1 -name 'fig_*.py' | head -1)
    [ -z "$script" ] && continue
    echo "  $(basename "$dir")"
    (cd "$dir" && python3 "$(basename "$script")")
done

cp Figures/Figure*/*.png figures_out/ 2>/dev/null || true
echo "── Done. Figures in figures_out/ ──"
