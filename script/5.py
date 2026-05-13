from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "datas" / "output"

INPUT_PATH = OUTPUT_DIR / "gasoline_cny_basket_40_80_change.csv"
OUTPUT_CSV_PATH = OUTPUT_DIR / "cny_cost_to_gasoline_price_ratio_time_series.csv"
OUTPUT_SVG_PATH = OUTPUT_DIR / "cny_cost_to_gasoline_price_ratio_time_series.svg"
SUMMARY_PATH = OUTPUT_DIR / "cny_cost_to_gasoline_price_ratio_time_series_summary.json"

COST_COLUMN = "理论国内成本_CNY_per_ton_本期10工作日均值"
GASOLINE_COLUMN = "汽油价格真实值_CNY_per_ton"
RATIO_COLUMN = "理论国内成本汽油价格比值"


def nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if math.isclose(vmin, vmax):
        return [vmin]

    raw_step = (vmax - vmin) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    normalized = raw_step / magnitude

    if normalized <= 1:
        step = 1 * magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude

    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def read_ratio_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["调整日期"] = pd.to_datetime(df["调整日期"], errors="coerce")
    df[COST_COLUMN] = pd.to_numeric(df[COST_COLUMN], errors="coerce")
    df[GASOLINE_COLUMN] = pd.to_numeric(df[GASOLINE_COLUMN], errors="coerce")
    df = df.dropna(subset=["调整日期", COST_COLUMN, GASOLINE_COLUMN])
    df = df[df[GASOLINE_COLUMN] != 0].copy()
    df[RATIO_COLUMN] = df[COST_COLUMN] / df[GASOLINE_COLUMN]
    return df.sort_values("调整日期").reset_index(drop=True)


def make_svg(df: pd.DataFrame) -> str:
    width = 1180
    height = 560
    left = 86
    right = 42
    top = 58
    bottom = 72
    chart_width = width - left - right
    chart_height = height - top - bottom

    if df.empty:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="30">No data</text></svg>'

    dates = df["调整日期"]
    x_values = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    y_values = df[RATIO_COLUMN].to_numpy(dtype=float)
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if math.isclose(x_min, x_max):
        x_max = x_min + 1

    y_ticks = nice_ticks(float(np.nanmin(y_values)), float(np.nanmax(y_values)))
    y_min, y_max = y_ticks[0], y_ticks[-1]

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * chart_width

    def y_scale(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_height

    path_commands = []
    point_elements = []
    for x_value, y_value, date in zip(x_values, y_values, dates, strict=True):
        command = "M" if not path_commands else "L"
        x = x_scale(float(x_value))
        y = y_scale(float(y_value))
        path_commands.append(f"{command}{x:.1f},{y:.1f}")
        point_elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2">'
            f'<title>{date.strftime("%Y-%m-%d")}  {y_value:.4f}</title>'
            f"</circle>"
        )

    time_ticks = pd.date_range(dates.min(), dates.max(), periods=min(8, len(df)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".line{fill:none;stroke:#2f6fbb;stroke-width:2}",
        "circle{fill:#2f6fbb;opacity:.85}",
        ".note{fill:#666;font-size:11px}",
        "</style>",
        f'<text class="title" x="{left}" y="26">理论国内成本10工作日均值 / 汽油价格真实值</text>',
        f'<text class="note" x="{left}" y="45">单位同为 CNY/ton，图中比值越高表示理论成本占汽油价格比例越高。</text>',
    ]

    for tick in y_ticks:
        y = y_scale(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')

    for tick in time_ticks:
        x = x_scale(float(tick.toordinal()))
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 48}" text-anchor="middle">{tick.strftime("%Y-%m")}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<path class="line" d="{html.escape(" ".join(path_commands))}"/>',
            *point_elements,
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_outputs(df: pd.DataFrame) -> None:
    output = df[
        [
            "调整日期",
            COST_COLUMN,
            GASOLINE_COLUMN,
            RATIO_COLUMN,
        ]
    ].sort_values("调整日期", ascending=False).copy()
    output["调整日期"] = output["调整日期"].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    OUTPUT_SVG_PATH.write_text(make_svg(df), encoding="utf-8")

    summary = {
        "input_file": str(INPUT_PATH),
        "output_csv": str(OUTPUT_CSV_PATH),
        "output_svg": str(OUTPUT_SVG_PATH),
        "formula": f"{RATIO_COLUMN} = {COST_COLUMN} / {GASOLINE_COLUMN}",
        "rows": int(len(df)),
        "date_min": df["调整日期"].min().strftime("%Y-%m-%d") if not df.empty else None,
        "date_max": df["调整日期"].max().strftime("%Y-%m-%d") if not df.empty else None,
        "ratio_min": float(df[RATIO_COLUMN].min()) if not df.empty else None,
        "ratio_max": float(df[RATIO_COLUMN].max()) if not df.empty else None,
        "ratio_mean": float(df[RATIO_COLUMN].mean()) if not df.empty else None,
        "ratio_median": float(df[RATIO_COLUMN].median()) if not df.empty else None,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    df = read_ratio_data()
    write_outputs(df)
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(f"Date range: {df['调整日期'].min().date()} to {df['调整日期'].max().date()}")
        print(f"Ratio min/max: {df[RATIO_COLUMN].min():.6f} / {df[RATIO_COLUMN].max():.6f}")
    print(f"Saved CSV: {OUTPUT_CSV_PATH}")
    print(f"Saved SVG: {OUTPUT_SVG_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
