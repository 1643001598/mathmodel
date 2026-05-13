from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "datas" / "output"

INPUT_PATH = OUTPUT_DIR / "gasoline_cost_10workday_change.csv"
FULL_STATS_PATH = OUTPUT_DIR / "ratio_bin_count_0_1.csv"
NONZERO_STATS_PATH = OUTPUT_DIR / "ratio_bin_count_0_1_nonzero.csv"
SVG_PATH = OUTPUT_DIR / "ratio_bin_count_0_1.svg"
SUMMARY_PATH = OUTPUT_DIR / "ratio_bin_count_0_1_summary.json"

RATIO_COLUMN = "汽油变化率_理论国内成本变化率比值"
BIN_WIDTH = 0.1


def load_ratio() -> pd.Series:
    df = pd.read_csv(INPUT_PATH)
    ratio = pd.to_numeric(df[RATIO_COLUMN], errors="coerce")
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return ratio


def build_bin_counts(ratio: pd.Series) -> pd.DataFrame:
    start = math.floor(ratio.min() / BIN_WIDTH) * BIN_WIDTH
    stop = math.ceil(ratio.max() / BIN_WIDTH) * BIN_WIDTH
    edges = np.round(np.arange(start, stop + BIN_WIDTH * 1.5, BIN_WIDTH), 10)
    counts, edges = np.histogram(ratio.to_numpy(), bins=edges)

    stats = pd.DataFrame(
        {
            "区间左端": edges[:-1],
            "区间右端": edges[1:],
            "数量": counts,
        }
    )
    stats["区间"] = stats.apply(
        lambda row: f"[{row['区间左端']:.1f}, {row['区间右端']:.1f})",
        axis=1,
    )
    stats.loc[stats.index[-1], "区间"] = (
        f"[{stats.iloc[-1]['区间左端']:.1f}, {stats.iloc[-1]['区间右端']:.1f}]"
    )
    stats["占比"] = stats["数量"] / len(ratio)
    return stats[["区间", "区间左端", "区间右端", "数量", "占比"]]


def save_stats(stats: pd.DataFrame) -> pd.DataFrame:
    nonzero = stats[stats["数量"] > 0].copy()
    stats.to_csv(FULL_STATS_PATH, index=False, encoding="utf-8-sig")
    nonzero.to_csv(NONZERO_STATS_PATH, index=False, encoding="utf-8-sig")
    return nonzero


def make_svg(nonzero: pd.DataFrame) -> str:
    margin_left = 110
    margin_right = 70
    margin_top = 56
    margin_bottom = 42
    chart_width = 760
    row_height = 22
    bar_height = 14
    height = margin_top + margin_bottom + row_height * len(nonzero)
    width = margin_left + chart_width + margin_right
    max_count = max(int(nonzero["数量"].max()), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".bar{fill:#2f6fbb}",
        ".bar.neg{fill:#c35a4a}",
        ".note{fill:#666;font-size:11px}",
        "</style>",
        f'<text class="title" x="{margin_left}" y="26">汽油变化率 / 理论国内成本变化率：0.1区间数量统计</text>',
        f'<text class="note" x="{margin_left}" y="45">仅绘制数量大于0的区间；完整区间统计见 CSV。</text>',
    ]

    for tick in range(0, max_count + 1):
        x = margin_left + chart_width * tick / max_count
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top - 6}" x2="{x:.1f}" y2="{height - margin_bottom + 4}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle">{tick}</text>')

    parts.append(f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom + 4}" x2="{margin_left + chart_width}" y2="{height - margin_bottom + 4}"/>')

    for idx, row in nonzero.reset_index(drop=True).iterrows():
        y = margin_top + idx * row_height
        count = int(row["数量"])
        bar_width = chart_width * count / max_count
        label = html.escape(str(row["区间"]))
        klass = "bar neg" if row["区间右端"] <= 0 else "bar"

        parts.append(f'<text x="{margin_left - 8}" y="{y + 12}" text-anchor="end">{label}</text>')
        parts.append(f'<rect class="{klass}" x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="2"/>')
        parts.append(f'<text x="{margin_left + bar_width + 6:.1f}" y="{y + 12}">{count}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def write_summary(ratio: pd.Series, stats: pd.DataFrame, nonzero: pd.DataFrame) -> None:
    summary = {
        "input_file": str(INPUT_PATH),
        "ratio_column": RATIO_COLUMN,
        "bin_width": BIN_WIDTH,
        "valid_ratio_count": int(len(ratio)),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "ratio_mean": float(ratio.mean()),
        "ratio_median": float(ratio.median()),
        "full_bin_count": int(len(stats)),
        "nonzero_bin_count": int(len(nonzero)),
        "outputs": {
            "full_stats": str(FULL_STATS_PATH),
            "nonzero_stats": str(NONZERO_STATS_PATH),
            "svg": str(SVG_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ratio = load_ratio()
    stats = build_bin_counts(ratio)
    nonzero = save_stats(stats)
    SVG_PATH.write_text(make_svg(nonzero), encoding="utf-8")
    write_summary(ratio, stats, nonzero)

    print(f"Valid ratio count: {len(ratio)}")
    print(f"Min: {ratio.min():.6f}")
    print(f"Max: {ratio.max():.6f}")
    print(f"Saved full stats: {FULL_STATS_PATH}")
    print(f"Saved nonzero stats: {NONZERO_STATS_PATH}")
    print(f"Saved chart: {SVG_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
