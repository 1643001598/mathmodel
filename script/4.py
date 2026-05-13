from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datas"
OUTPUT_DIR = DATA_DIR / "output"

PRODUCT_OIL_PATH = DATA_DIR / "product_oil.csv"
BASKET_COST_PATH = OUTPUT_DIR / "basket_cost_model_ready.csv"

USD_TABLE_PATH = OUTPUT_DIR / "gasoline_usd_basket_40_80_change.csv"
CNY_TABLE_PATH = OUTPUT_DIR / "gasoline_cny_basket_40_80_change.csv"
ALL_ALIGNED_PATH = OUTPUT_DIR / "gasoline_basket_40_80_all_aligned.csv"

USD_BIN_FULL_PATH = OUTPUT_DIR / "usd_40_80_ratio_bin_count_0_1.csv"
USD_BIN_NONZERO_PATH = OUTPUT_DIR / "usd_40_80_ratio_bin_count_0_1_nonzero.csv"
USD_BIN_SVG_PATH = OUTPUT_DIR / "usd_40_80_ratio_bin_count_0_1.svg"
USD_LINE_SVG_PATH = OUTPUT_DIR / "usd_40_80_cost_gasoline_time_series.svg"

CNY_BIN_FULL_PATH = OUTPUT_DIR / "cny_40_80_ratio_bin_count_0_1.csv"
CNY_BIN_NONZERO_PATH = OUTPUT_DIR / "cny_40_80_ratio_bin_count_0_1_nonzero.csv"
CNY_BIN_SVG_PATH = OUTPUT_DIR / "cny_40_80_ratio_bin_count_0_1.svg"
CNY_LINE_SVG_PATH = OUTPUT_DIR / "cny_40_80_cost_gasoline_time_series.svg"

SUMMARY_PATH = OUTPUT_DIR / "usd_cny_40_80_analysis_summary.json"

ROLLING_WORKDAYS = 10
ALIGN_TOLERANCE_DAYS = 10
START_DATE = pd.Timestamp("2016-01-01")
LOWER_USD_PER_BBL = 40.0
UPPER_USD_PER_BBL = 80.0
BARRELS_PER_TON = 7.33
BIN_WIDTH = 0.1


def parse_price(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    multiplier = 10000 if "万" in text else 1
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group()) * multiplier


def parse_change(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    sign = -1 if "↓" in text or "-" in text else 1
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return sign * float(match.group())


def read_gasoline() -> pd.DataFrame:
    product = pd.read_csv(PRODUCT_OIL_PATH)
    gasoline = product[["调整日期", "汽油", "汽油涨跌"]].copy()
    gasoline["调整日期"] = pd.to_datetime(gasoline["调整日期"], errors="coerce")
    gasoline["汽油价格真实值_CNY_per_ton"] = gasoline["汽油"].map(parse_price)
    gasoline["汽油涨跌真实值_CNY_per_ton"] = gasoline["汽油涨跌"].map(parse_change)
    gasoline = gasoline.dropna(subset=["调整日期", "汽油价格真实值_CNY_per_ton"])
    return gasoline.sort_values("调整日期").reset_index(drop=True)


def read_basket_cost() -> pd.DataFrame:
    cost = pd.read_csv(BASKET_COST_PATH)
    cost["日期"] = pd.to_datetime(cost["日期"], errors="coerce")
    cost["一揽子价格_USD_per_bbl"] = pd.to_numeric(
        cost["一揽子价格_USD_per_bbl"],
        errors="coerce",
    )
    cost["USD_CNY"] = pd.to_numeric(cost["USD_CNY"], errors="coerce")
    cost = cost.dropna(subset=["日期", "一揽子价格_USD_per_bbl", "USD_CNY"])
    cost = cost.sort_values("日期").reset_index(drop=True)

    cost["理论国内成本_USD_per_bbl_本期10工作日均值"] = (
        cost["一揽子价格_USD_per_bbl"]
        .rolling(window=ROLLING_WORKDAYS, min_periods=ROLLING_WORKDAYS)
        .mean()
    )
    cost["计算日汇率_USD_CNY"] = cost["USD_CNY"]
    cost["理论国内成本_CNY_per_ton_本期10工作日均值"] = (
        cost["理论国内成本_USD_per_bbl_本期10工作日均值"]
        * cost["计算日汇率_USD_CNY"]
        * BARRELS_PER_TON
    )
    cost["10工作日窗口开始日期"] = cost["日期"].shift(ROLLING_WORKDAYS - 1)
    cost["10工作日窗口结束日期"] = cost["日期"]
    return cost


def align_cost_to_adjustments(
    gasoline: pd.DataFrame,
    cost: pd.DataFrame,
) -> pd.DataFrame:
    cost_for_merge = cost[
        [
            "日期",
            "一揽子价格_USD_per_bbl",
            "计算日汇率_USD_CNY",
            "理论国内成本_USD_per_bbl_本期10工作日均值",
            "理论国内成本_CNY_per_ton_本期10工作日均值",
            "10工作日窗口开始日期",
            "10工作日窗口结束日期",
        ]
    ].dropna(subset=["理论国内成本_USD_per_bbl_本期10工作日均值"])

    aligned = pd.merge_asof(
        gasoline.sort_values("调整日期"),
        cost_for_merge.sort_values("日期"),
        left_on="调整日期",
        right_on="日期",
        direction="backward",
        tolerance=pd.Timedelta(days=ALIGN_TOLERANCE_DAYS),
    )
    return aligned.rename(columns={"日期": "匹配成本日期"})


def add_change_rates(aligned: pd.DataFrame) -> pd.DataFrame:
    result = aligned.sort_values("调整日期").copy()

    result["理论国内成本_USD_per_bbl_上期10工作日均值"] = result[
        "理论国内成本_USD_per_bbl_本期10工作日均值"
    ].shift(1)
    result["理论国内成本_USD_per_bbl_涨跌值"] = (
        result["理论国内成本_USD_per_bbl_本期10工作日均值"]
        - result["理论国内成本_USD_per_bbl_上期10工作日均值"]
    )
    result["理论国内成本_USD_per_bbl_变化率"] = (
        result["理论国内成本_USD_per_bbl_本期10工作日均值"]
        / result["理论国内成本_USD_per_bbl_上期10工作日均值"]
        - 1
    )

    result["理论国内成本_CNY_per_ton_上期10工作日均值"] = result[
        "理论国内成本_CNY_per_ton_本期10工作日均值"
    ].shift(1)
    result["理论国内成本_CNY_per_ton_涨跌值"] = (
        result["理论国内成本_CNY_per_ton_本期10工作日均值"]
        - result["理论国内成本_CNY_per_ton_上期10工作日均值"]
    )
    result["理论国内成本_CNY_per_ton_变化率"] = (
        result["理论国内成本_CNY_per_ton_本期10工作日均值"]
        / result["理论国内成本_CNY_per_ton_上期10工作日均值"]
        - 1
    )

    result["上期汽油价格真实值_CNY_per_ton"] = result["汽油价格真实值_CNY_per_ton"].shift(1)
    result["汽油价格变化率"] = (
        result["汽油价格真实值_CNY_per_ton"]
        / result["上期汽油价格真实值_CNY_per_ton"]
        - 1
    )

    usd_denominator = result["理论国内成本_USD_per_bbl_变化率"].replace(0, pd.NA)
    cny_denominator = result["理论国内成本_CNY_per_ton_变化率"].replace(0, pd.NA)
    result["汽油变化率_理论国内成本USD变化率比值"] = result["汽油价格变化率"] / usd_denominator
    result["汽油变化率_理论国内成本CNY变化率比值"] = result["汽油价格变化率"] / cny_denominator
    return result


def filter_analysis_rows(result: pd.DataFrame) -> pd.DataFrame:
    after_start_date = result["调整日期"] >= START_DATE
    current_in_range = result["理论国内成本_USD_per_bbl_本期10工作日均值"].between(
        LOWER_USD_PER_BBL,
        UPPER_USD_PER_BBL,
        inclusive="both",
    )
    previous_in_range = result["理论国内成本_USD_per_bbl_上期10工作日均值"].between(
        LOWER_USD_PER_BBL,
        UPPER_USD_PER_BBL,
        inclusive="both",
    )
    return result[after_start_date & current_in_range & previous_in_range].copy()


def usd_display_columns() -> list[str]:
    return [
        "调整日期",
        "汽油价格真实值_CNY_per_ton",
        "汽油涨跌真实值_CNY_per_ton",
        "理论国内成本_USD_per_bbl_本期10工作日均值",
        "理论国内成本_USD_per_bbl_上期10工作日均值",
        "理论国内成本_USD_per_bbl_涨跌值",
        "汽油价格变化率",
        "理论国内成本_USD_per_bbl_变化率",
        "汽油变化率_理论国内成本USD变化率比值",
        "匹配成本日期",
        "10工作日窗口开始日期",
        "10工作日窗口结束日期",
    ]


def cny_display_columns() -> list[str]:
    return [
        "调整日期",
        "汽油价格真实值_CNY_per_ton",
        "汽油涨跌真实值_CNY_per_ton",
        "计算日汇率_USD_CNY",
        "理论国内成本_USD_per_bbl_本期10工作日均值",
        "理论国内成本_USD_per_bbl_上期10工作日均值",
        "理论国内成本_CNY_per_ton_本期10工作日均值",
        "理论国内成本_CNY_per_ton_上期10工作日均值",
        "理论国内成本_CNY_per_ton_涨跌值",
        "汽油价格变化率",
        "理论国内成本_CNY_per_ton_变化率",
        "汽油变化率_理论国内成本CNY变化率比值",
        "匹配成本日期",
        "10工作日窗口开始日期",
        "10工作日窗口结束日期",
    ]


def format_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ["调整日期", "匹配成本日期", "10工作日窗口开始日期", "10工作日窗口结束日期"]:
        formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
    return formatted


def clean_ratio(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def build_bin_counts(values: pd.Series) -> pd.DataFrame:
    clean = clean_ratio(values)
    if clean.empty:
        return pd.DataFrame(columns=["区间", "区间左端", "区间右端", "数量", "占比"])

    start = math.floor(clean.min() / BIN_WIDTH) * BIN_WIDTH
    stop = math.ceil(clean.max() / BIN_WIDTH) * BIN_WIDTH
    edges = np.round(np.arange(start, stop + BIN_WIDTH * 1.5, BIN_WIDTH), 10)
    counts, edges = np.histogram(clean.to_numpy(), bins=edges)

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
    stats["占比"] = stats["数量"] / len(clean)
    return stats[["区间", "区间左端", "区间右端", "数量", "占比"]]


def make_bar_svg(nonzero: pd.DataFrame, title: str, note: str) -> str:
    margin_left = 110
    margin_right = 70
    margin_top = 56
    margin_bottom = 42
    chart_width = 760
    row_height = 22
    bar_height = 14
    height = margin_top + margin_bottom + row_height * max(len(nonzero), 1)
    width = margin_left + chart_width + margin_right
    max_count = max(int(nonzero["数量"].max()), 1) if not nonzero.empty else 1

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
        f'<text class="title" x="{margin_left}" y="26">{html.escape(title)}</text>',
        f'<text class="note" x="{margin_left}" y="45">{html.escape(note)}</text>',
    ]

    for tick in range(0, max_count + 1):
        x = margin_left + chart_width * tick / max_count
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{margin_top - 6}" x2="{x:.1f}" y2="{height - margin_bottom + 4}"/>'
        )
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle">{tick}</text>')

    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom + 4}" x2="{margin_left + chart_width}" y2="{height - margin_bottom + 4}"/>'
    )

    for idx, row in nonzero.reset_index(drop=True).iterrows():
        y = margin_top + idx * row_height
        count = int(row["数量"])
        bar_width = chart_width * count / max_count
        label = html.escape(str(row["区间"]))
        klass = "bar neg" if row["区间右端"] <= 0 else "bar"
        parts.append(f'<text x="{margin_left - 8}" y="{y + 12}" text-anchor="end">{label}</text>')
        parts.append(
            f'<rect class="{klass}" x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="2"/>'
        )
        parts.append(f'<text x="{margin_left + bar_width + 6:.1f}" y="{y + 12}">{count}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


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


def make_line_svg(
    filtered: pd.DataFrame,
    cost_column: str,
    cost_label: str,
    left_axis_label: str,
    title: str,
) -> str:
    data = filtered.dropna(
        subset=["调整日期", cost_column, "汽油价格真实值_CNY_per_ton"]
    ).sort_values("调整日期")

    width = 1180
    height = 560
    left = 96
    right = 96
    top = 58
    bottom = 72
    chart_width = width - left - right
    chart_height = height - top - bottom

    if data.empty:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="30">No data</text></svg>'

    dates = data["调整日期"]
    x_values = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if math.isclose(x_min, x_max):
        x_max = x_min + 1

    cost = data[cost_column].to_numpy(dtype=float)
    gasoline = data["汽油价格真实值_CNY_per_ton"].to_numpy(dtype=float)
    cost_ticks = nice_ticks(float(np.nanmin(cost)), float(np.nanmax(cost)))
    gasoline_ticks = nice_ticks(float(np.nanmin(gasoline)), float(np.nanmax(gasoline)))
    cost_min, cost_max = cost_ticks[0], cost_ticks[-1]
    gas_min, gas_max = gasoline_ticks[0], gasoline_ticks[-1]

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * chart_width

    def y_cost(value: float) -> float:
        return top + (cost_max - value) / (cost_max - cost_min) * chart_height

    def y_gas(value: float) -> float:
        return top + (gas_max - value) / (gas_max - gas_min) * chart_height

    def path_for(values: np.ndarray, y_func) -> str:
        commands = []
        for x_value, y_value in zip(x_values, values, strict=True):
            command = "M" if not commands else "L"
            commands.append(f"{command}{x_scale(float(x_value)):.1f},{y_func(float(y_value)):.1f}")
        return " ".join(commands)

    time_ticks = pd.date_range(dates.min(), dates.max(), periods=min(8, len(data)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".cost{fill:none;stroke:#2f6fbb;stroke-width:2}",
        ".gas{fill:none;stroke:#c35a4a;stroke-width:2}",
        ".note{fill:#666;font-size:11px}",
        "</style>",
        f'<text class="title" x="{left}" y="26">{html.escape(title)}</text>',
        f'<text class="note" x="{left}" y="45">左轴：{html.escape(left_axis_label)}；右轴：汽油价格真实值（CNY/ton）。</text>',
    ]

    for tick in cost_ticks:
        y = y_cost(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')

    for tick in gasoline_ticks:
        y = y_gas(tick)
        parts.append(f'<text x="{width - right + 8}" y="{y + 4:.1f}">{tick:g}</text>')

    for tick in time_ticks:
        x = x_scale(float(tick.toordinal()))
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 48}" text-anchor="middle">{tick.strftime("%Y-%m")}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{width - right}" y1="{top}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<path class="cost" d="{path_for(cost, y_cost)}"/>',
            f'<path class="gas" d="{path_for(gasoline, y_gas)}"/>',
            f'<rect x="{left}" y="{height - 32}" width="18" height="4" fill="#2f6fbb"/>',
            f'<text x="{left + 26}" y="{height - 26}">{html.escape(cost_label)}</text>',
            f'<rect x="{left + 390}" y="{height - 32}" width="18" height="4" fill="#c35a4a"/>',
            f'<text x="{left + 416}" y="{height - 26}">汽油价格真实值（CNY/ton）</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts)


def write_table(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    display = df.sort_values("调整日期", ascending=False)[columns].copy()
    format_date_columns(display).to_csv(path, index=False, encoding="utf-8-sig")


def write_bin_outputs(
    filtered: pd.DataFrame,
    ratio_column: str,
    full_path: Path,
    nonzero_path: Path,
    svg_path: Path,
    title: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = build_bin_counts(filtered[ratio_column])
    nonzero = stats[stats["数量"] > 0].copy()
    stats.to_csv(full_path, index=False, encoding="utf-8-sig")
    nonzero.to_csv(nonzero_path, index=False, encoding="utf-8-sig")
    svg_path.write_text(
        make_bar_svg(nonzero, title, "0.1为分箱宽度，仅绘制数量大于0的区间。"),
        encoding="utf-8",
    )
    return stats, nonzero


def write_outputs(aligned: pd.DataFrame, filtered: pd.DataFrame) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_table(aligned, ALL_ALIGNED_PATH, cny_display_columns())
    write_table(filtered, USD_TABLE_PATH, usd_display_columns())
    write_table(filtered, CNY_TABLE_PATH, cny_display_columns())

    usd_stats, usd_nonzero = write_bin_outputs(
        filtered,
        "汽油变化率_理论国内成本USD变化率比值",
        USD_BIN_FULL_PATH,
        USD_BIN_NONZERO_PATH,
        USD_BIN_SVG_PATH,
        "40-80美元/桶区间：汽油变化率 / 理论成本USD变化率",
    )
    cny_stats, cny_nonzero = write_bin_outputs(
        filtered,
        "汽油变化率_理论国内成本CNY变化率比值",
        CNY_BIN_FULL_PATH,
        CNY_BIN_NONZERO_PATH,
        CNY_BIN_SVG_PATH,
        "40-80美元/桶区间：汽油变化率 / 理论成本CNY变化率",
    )

    USD_LINE_SVG_PATH.write_text(
        make_line_svg(
            filtered,
            "理论国内成本_USD_per_bbl_本期10工作日均值",
            "理论国内成本10工作日均值（USD/bbl）",
            "理论国内成本10工作日均值（USD/bbl）",
            "40-80美元/桶区间：理论成本USD口径与汽油价格",
        ),
        encoding="utf-8",
    )
    CNY_LINE_SVG_PATH.write_text(
        make_line_svg(
            filtered,
            "理论国内成本_CNY_per_ton_本期10工作日均值",
            "理论国内成本10工作日均值（CNY/ton）",
            "理论国内成本10工作日均值（CNY/ton）",
            "40-80美元/桶区间：理论成本CNY口径与汽油价格",
        ),
        encoding="utf-8",
    )

    return {
        "usd_stats": usd_stats,
        "usd_nonzero": usd_nonzero,
        "cny_stats": cny_stats,
        "cny_nonzero": cny_nonzero,
    }


def write_summary(aligned: pd.DataFrame, filtered: pd.DataFrame, outputs: dict[str, object]) -> None:
    usd_ratio = clean_ratio(filtered["汽油变化率_理论国内成本USD变化率比值"])
    cny_ratio = clean_ratio(filtered["汽油变化率_理论国内成本CNY变化率比值"])
    summary = {
        "input_product_file": str(PRODUCT_OIL_PATH),
        "input_basket_cost_file": str(BASKET_COST_PATH),
        "rolling_workdays": ROLLING_WORKDAYS,
        "barrels_per_ton": BARRELS_PER_TON,
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "filter_rule": (
            "keep rows on or after 2016-01-01 where current and previous "
            "10-workday average basket prices are both within [40, 80] USD/bbl"
        ),
        "cny_conversion_rule": (
            "theoretical_cost_cny_per_ton = "
            "10-workday-average basket USD/bbl * USD_CNY on matched calculation date * 7.33"
        ),
        "aligned_rows": int(len(aligned)),
        "filtered_rows": int(len(filtered)),
        "filtered_date_min": filtered["调整日期"].min().strftime("%Y-%m-%d") if not filtered.empty else None,
        "filtered_date_max": filtered["调整日期"].max().strftime("%Y-%m-%d") if not filtered.empty else None,
        "usd_ratio": {
            "valid_rows": int(len(usd_ratio)),
            "min": float(usd_ratio.min()) if not usd_ratio.empty else None,
            "max": float(usd_ratio.max()) if not usd_ratio.empty else None,
            "full_bin_count": int(len(outputs["usd_stats"])),
            "nonzero_bin_count": int(len(outputs["usd_nonzero"])),
        },
        "cny_ratio": {
            "valid_rows": int(len(cny_ratio)),
            "min": float(cny_ratio.min()) if not cny_ratio.empty else None,
            "max": float(cny_ratio.max()) if not cny_ratio.empty else None,
            "full_bin_count": int(len(outputs["cny_stats"])),
            "nonzero_bin_count": int(len(outputs["cny_nonzero"])),
        },
        "outputs": {
            "usd_filtered_table": str(USD_TABLE_PATH),
            "cny_filtered_table": str(CNY_TABLE_PATH),
            "all_aligned_table": str(ALL_ALIGNED_PATH),
            "usd_bin_full": str(USD_BIN_FULL_PATH),
            "usd_bin_nonzero": str(USD_BIN_NONZERO_PATH),
            "usd_bin_svg": str(USD_BIN_SVG_PATH),
            "usd_line_svg": str(USD_LINE_SVG_PATH),
            "cny_bin_full": str(CNY_BIN_FULL_PATH),
            "cny_bin_nonzero": str(CNY_BIN_NONZERO_PATH),
            "cny_bin_svg": str(CNY_BIN_SVG_PATH),
            "cny_line_svg": str(CNY_LINE_SVG_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    gasoline = read_gasoline()
    cost = read_basket_cost()
    aligned = add_change_rates(align_cost_to_adjustments(gasoline, cost))
    filtered = filter_analysis_rows(aligned)
    outputs = write_outputs(aligned, filtered)
    write_summary(aligned, filtered, outputs)

    cny_ratio = clean_ratio(filtered["汽油变化率_理论国内成本CNY变化率比值"])
    print(f"Aligned rows: {len(aligned)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"CNY valid ratio rows: {len(cny_ratio)}")
    if not cny_ratio.empty:
        print(f"CNY ratio min/max: {cny_ratio.min():.6f} / {cny_ratio.max():.6f}")
    print(f"Saved CNY table: {CNY_TABLE_PATH}")
    print(f"Saved CNY bin chart: {CNY_BIN_SVG_PATH}")
    print(f"Saved CNY line chart: {CNY_LINE_SVG_PATH}")


if __name__ == "__main__":
    main()
