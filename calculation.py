from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datas"
OUTPUT_DIR = DATA_DIR / "output2"
OUTPUT3_DIR = DATA_DIR / "output3"

PRODUCT_OIL_PATH = DATA_DIR / "product_oil.csv"
BRENT_PATH = DATA_DIR / "brent.csv"
DUBAI_PATH = DATA_DIR / "dubai.csv"
WTI_PATH = DATA_DIR / "wti.csv"
USD_CNY_PATH = DATA_DIR / "USD_CNY.csv"

OUTPUT_TABLE_PATH = OUTPUT_DIR / "product_oil_interval_averages.csv"
OUTPUT_CHART_PATH = OUTPUT_DIR / "product_oil_interval_averages.svg"
SUMMARY_PATH = OUTPUT_DIR / "product_oil_interval_averages_summary.json"
OUTPUT_TABLE_2016_PATH = OUTPUT_DIR / "product_oil_interval_averages_from_2016.csv"
OUTPUT_CHART_2016_PATH = OUTPUT_DIR / "product_oil_interval_averages_from_2016.svg"
SUMMARY_2016_PATH = OUTPUT_DIR / "product_oil_interval_averages_from_2016_summary.json"
RATIO_MODEL_SAMPLES_2016_PATH = OUTPUT_DIR / "oil_40_80_ratio_model_samples_from_2016.csv"
RATIO_MODEL_SUMMARY_2016_PATH = OUTPUT_DIR / "oil_40_80_ratio_model_summary_from_2016.csv"
RATIO_MODEL_JSON_2016_PATH = OUTPUT_DIR / "oil_40_80_ratio_model_summary_from_2016.json"
RATIO_MODEL_CHART_2016_PATH = OUTPUT_DIR / "oil_40_80_ratio_model_from_2016.svg"
RATIO_MODEL_STRICT_SAMPLES_2016_PATH = (
    OUTPUT_DIR / "oil_40_80_strict_all_series_samples_from_2016.csv"
)
LINEAR_MODEL_SAMPLES_2016_PATH = OUTPUT_DIR / "oil_40_80_linear_model_samples_from_2016.csv"
LINEAR_MODEL_SUMMARY_2016_PATH = OUTPUT_DIR / "oil_40_80_linear_model_summary_from_2016.csv"
LINEAR_MODEL_JSON_2016_PATH = OUTPUT_DIR / "oil_40_80_linear_model_summary_from_2016.json"
LINEAR_MODEL_CHART_2016_PATH = OUTPUT_DIR / "oil_40_80_linear_model_from_2016.svg"
UP_DOWN_ANALYZE_DIR = OUTPUT_DIR / "imgs" / "up_down_analyze"
UP_DOWN_MODEL_SUMMARY_2016_PATH = OUTPUT_DIR / "oil_40_80_up_down_linear_model_summary_from_2016.csv"
UP_DOWN_MODEL_SAMPLES_2016_PATH = OUTPUT_DIR / "oil_40_80_up_down_linear_model_samples_from_2016.csv"
UP_DOWN_MODEL_JSON_2016_PATH = OUTPUT_DIR / "oil_40_80_up_down_linear_model_summary_from_2016.json"
ADJUSTMENT_CHANGE_SAMPLES_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change_samples.csv"
)
ADJUSTMENT_CHANGE_SUMMARY_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change_summary.json"
)
ADJUSTMENT_CHANGE_CHART_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change.svg"
)
ADJUSTMENT_CHANGE_LINEAR_SAMPLES_2016_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change_linear_samples_from_2016.csv"
)
ADJUSTMENT_CHANGE_LINEAR_SUMMARY_2016_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change_linear_summary_from_2016.csv"
)
ADJUSTMENT_CHANGE_LINEAR_JSON_2016_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change_linear_summary_from_2016.json"
)
ADJUSTMENT_CHANGE_LINEAR_CHART_2016_PATH = (
    OUTPUT3_DIR / "domestic_adjustment_vs_international_change_linear_from_2016.svg"
)

START_DATE = pd.Timestamp("2000-06-06")
END_DATE = pd.Timestamp("2026-05-09")
FILTERED_START_DATE = pd.Timestamp("2016-01-01")

# 沿用本项目已有脚本中的换算口径：1 metric ton ~= 7.33 barrels。
BARRELS_PER_TON = 7.33
FX_MAX_STALE_DAYS = 7

OIL_SOURCES = {
    "布伦特": BRENT_PATH,
    "迪拜": DUBAI_PATH,
    "WTI": WTI_PATH,
}

MODEL_TARGETS = {
    "布伦特": "布伦特均价_USD_per_bbl",
    "迪拜": "迪拜均价_USD_per_bbl",
    "WTI": "WTI均价_USD_per_bbl",
    "三油算术均价": "三油算术均价_USD_per_bbl",
}


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


def read_product_oil() -> pd.DataFrame:
    product = pd.read_csv(PRODUCT_OIL_PATH)
    product["调整日期"] = pd.to_datetime(product["调整日期"], errors="coerce")
    product["汽油价格_CNY_per_ton"] = product["汽油"].map(parse_price)
    product["国内调价幅度_CNY_per_ton"] = product["汽油涨跌"].map(parse_change)
    product = product.dropna(subset=["调整日期", "汽油价格_CNY_per_ton"])
    product = product[
        product["调整日期"].between(START_DATE, END_DATE, inclusive="both")
    ].copy()
    return product.sort_values("调整日期").reset_index(drop=True)


def read_usd_cny() -> pd.DataFrame:
    fx = pd.read_csv(USD_CNY_PATH)
    fx["汇率日期"] = pd.to_datetime(fx["日期"], errors="coerce")
    fx["USD_CNY"] = pd.to_numeric(fx["收盘"], errors="coerce")
    fx = fx.dropna(subset=["汇率日期", "USD_CNY"])
    return fx[["汇率日期", "USD_CNY"]].sort_values("汇率日期").reset_index(drop=True)


def add_product_usd_per_bbl(product: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge_asof(
        product.sort_values("调整日期"),
        fx.sort_values("汇率日期"),
        left_on="调整日期",
        right_on="汇率日期",
        direction="backward",
        tolerance=pd.Timedelta(days=FX_MAX_STALE_DAYS),
    )
    result["汇率间隔天数"] = (result["调整日期"] - result["汇率日期"]).dt.days
    result["汽油价格_USD_per_bbl"] = (
        result["汽油价格_CNY_per_ton"] / result["USD_CNY"] / BARRELS_PER_TON
    )
    return result


def read_oil_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"Date", "Price"}.issubset(df.columns):
        result = df.rename(columns={"Date": "日期", "Price": "价格"})[["日期", "价格"]]
    elif {"日期", "收盘"}.issubset(df.columns):
        result = df.rename(columns={"收盘": "价格"})[["日期", "价格"]]
    else:
        result = df.iloc[:, [0, 1]].copy()
        result.columns = ["日期", "价格"]

    result["日期"] = pd.to_datetime(result["日期"], errors="coerce")
    result["价格"] = pd.to_numeric(result["价格"], errors="coerce")
    result = result.dropna(subset=["日期", "价格"])
    result = result[result["日期"].between(START_DATE, END_DATE, inclusive="both")]
    return result.sort_values("日期").reset_index(drop=True)


def interval_mean(
    prices: pd.DataFrame,
    previous_date: pd.Timestamp | pd.NaT,
    current_date: pd.Timestamp,
) -> tuple[float | None, int]:
    if pd.isna(previous_date):
        return None, 0

    in_window = prices[(prices["日期"] > previous_date) & (prices["日期"] <= current_date)]
    if in_window.empty:
        return None, 0
    return float(in_window["价格"].mean()), int(len(in_window))


def add_interval_averages(
    product: pd.DataFrame,
    oil_prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = product.copy()
    result["上次调整日期"] = result["调整日期"].shift(1)

    for name, prices in oil_prices.items():
        averages: list[float | None] = []
        counts: list[int] = []
        for row in result.itertuples(index=False):
            mean_value, count = interval_mean(prices, row.上次调整日期, row.调整日期)
            averages.append(mean_value)
            counts.append(count)
        result[f"{name}均价_USD_per_bbl"] = averages
        result[f"{name}样本天数"] = counts

    mean_columns = [f"{name}均价_USD_per_bbl" for name in oil_prices]
    result["三油算术均价_USD_per_bbl"] = result[mean_columns].mean(axis=1, skipna=True)
    result["三油可用数量"] = result[mean_columns].notna().sum(axis=1)
    return result


def format_dates_for_output(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ["调整日期", "上次调整日期", "汇率日期"]:
        if column in result.columns:
            result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if math.isclose(vmin, vmax):
        return [vmin]

    raw_step = (vmax - vmin) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
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


def svg_path_for_series(
    data: pd.DataFrame,
    column: str,
    x_scale,
    y_scale,
) -> str:
    commands: list[str] = []
    drawing = False
    for row in data[["x_value", column]].itertuples(index=False):
        x_value = float(row.x_value)
        y_value = row[1]
        if pd.isna(y_value):
            drawing = False
            continue

        command = "L" if drawing else "M"
        commands.append(f"{command}{x_scale(x_value):.1f},{y_scale(float(y_value)):.1f}")
        drawing = True
    return " ".join(commands)


def make_line_svg(result: pd.DataFrame) -> str:
    columns = [
        ("汽油价格_USD_per_bbl", "汽油(USD/bbl)", "#c35a4a"),
        ("布伦特均价_USD_per_bbl", "布伦特区间均价", "#2f6fbb"),
        ("迪拜均价_USD_per_bbl", "迪拜区间均价", "#7a5dbb"),
        ("WTI均价_USD_per_bbl", "WTI区间均价", "#348f6c"),
        ("三油算术均价_USD_per_bbl", "三油算术均价", "#222222"),
    ]
    data = result.dropna(subset=["调整日期"]).sort_values("调整日期").copy()
    data["x_value"] = data["调整日期"].map(pd.Timestamp.toordinal).astype(float)

    width = 1280
    height = 620
    left = 92
    right = 52
    top = 66
    bottom = 96
    chart_width = width - left - right
    chart_height = height - top - bottom

    values = pd.concat([data[column] for column, _, _ in columns], ignore_index=True)
    values = values.dropna()
    if data.empty or values.empty:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<text x="20" y="30">No data</text></svg>'
        )

    x_min = float(data["x_value"].min())
    x_max = float(data["x_value"].max())
    if math.isclose(x_min, x_max):
        x_max = x_min + 1

    y_ticks = nice_ticks(float(values.min()), float(values.max()))
    y_min = y_ticks[0]
    y_max = y_ticks[-1]
    if math.isclose(y_min, y_max):
        y_max = y_min + 1

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * chart_width

    def y_scale(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_height

    time_ticks = pd.date_range(
        data["调整日期"].min(),
        data["调整日期"].max(),
        periods=min(9, len(data)),
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".line{fill:none;stroke-width:2}",
        "</style>",
        f'<text class="title" x="{left}" y="30">成品油调价日与国际原油区间均价</text>',
        (
            f'<text class="note" x="{left}" y="50">'
            f"区间均值按 (prev, cur] 计算；汽油价格按 USD_CNY 与 {BARRELS_PER_TON:g} bbl/ton 换算为 USD/bbl。"
            "</text>"
        ),
    ]

    for tick in y_ticks:
        y = y_scale(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')

    for tick in time_ticks:
        x = x_scale(float(tick.toordinal()))
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 58}" text-anchor="middle">{tick.strftime("%Y-%m")}</text>'
        )

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<text x="{left}" y="{height - 24}">单位：USD/bbl</text>',
        ]
    )

    for column, label, color in columns:
        path = svg_path_for_series(data, column, x_scale, y_scale)
        if path:
            parts.append(f'<path class="line" d="{path}" stroke="{color}"/>')

    legend_x = left + 180
    legend_y = height - 30
    for idx, (_, label, color) in enumerate(columns):
        x = legend_x + idx * 190
        parts.append(f'<rect x="{x}" y="{legend_y - 8}" width="18" height="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 26}" y="{legend_y - 4}">{html.escape(label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def fit_scale_model(sample: pd.DataFrame, target_column: str) -> dict[str, float | int | None]:
    clean = sample.dropna(subset=["汽油价格_USD_per_bbl", target_column]).copy()
    if clean.empty:
        return {
            "样本数": 0,
            "国内汽油均值_USD_per_bbl": None,
            "国际原油均值_USD_per_bbl": None,
            "平均国内汽油_国际原油比值": None,
            "中位数国内汽油_国际原油比值": None,
            "缩放系数_国际原油等于系数乘国内汽油": None,
            "缩放模型RMSE_USD_per_bbl": None,
            "缩放模型MAE_USD_per_bbl": None,
            "缩放模型R2": None,
        }

    x = clean["汽油价格_USD_per_bbl"].astype(float)
    y = clean[target_column].astype(float)
    denominator = float((x * x).sum())
    scale = float((x * y).sum() / denominator) if denominator else None

    if scale is None:
        predicted = x * 0
        rmse = None
        mae = None
        r2 = None
    else:
        predicted = x * scale
        residual = y - predicted
        rmse = float(math.sqrt((residual * residual).mean()))
        mae = float(residual.abs().mean())
        total = float(((y - y.mean()) * (y - y.mean())).sum())
        squared_error = float((residual * residual).sum())
        r2 = float(1 - squared_error / total) if total else None

    return {
        "样本数": int(len(clean)),
        "国内汽油均值_USD_per_bbl": float(x.mean()),
        "国际原油均值_USD_per_bbl": float(y.mean()),
        "平均国内汽油_国际原油比值": float((x / y).mean()),
        "中位数国内汽油_国际原油比值": float((x / y).median()),
        "缩放系数_国际原油等于系数乘国内汽油": scale,
        "缩放模型RMSE_USD_per_bbl": rmse,
        "缩放模型MAE_USD_per_bbl": mae,
        "缩放模型R2": r2,
    }


def build_ratio_model_outputs(result_2016: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []

    for target_name, target_column in MODEL_TARGETS.items():
        sample = result_2016[
            result_2016[target_column].between(40, 80, inclusive="both")
        ].copy()
        stats = fit_scale_model(sample, target_column)
        summary_rows.append(
            {
                "油价口径": target_name,
                "筛选规则": f"{target_name}区间均价在[40,80] USD/bbl",
                "国际原油列": target_column,
                **stats,
            }
        )

        if not sample.empty:
            scale = stats["缩放系数_国际原油等于系数乘国内汽油"]
            sample["油价口径"] = target_name
            sample["国际原油均价_USD_per_bbl"] = sample[target_column]
            sample["国内汽油_国际原油比值"] = (
                sample["汽油价格_USD_per_bbl"] / sample["国际原油均价_USD_per_bbl"]
            )
            sample["缩放拟合国际原油_USD_per_bbl"] = (
                sample["汽油价格_USD_per_bbl"] * float(scale)
                if scale is not None
                else pd.NA
            )
            sample["缩放拟合残差_USD_per_bbl"] = (
                sample["国际原油均价_USD_per_bbl"] - sample["缩放拟合国际原油_USD_per_bbl"]
            )
            sample_frames.append(sample)

    strict_columns = list(MODEL_TARGETS.values())
    strict = result_2016.dropna(subset=["汽油价格_USD_per_bbl", *strict_columns]).copy()
    strict = strict[
        strict[strict_columns].apply(lambda row: row.between(40, 80).all(), axis=1)
    ].copy()
    strict_stats = fit_scale_model(strict, "三油算术均价_USD_per_bbl")
    summary_rows.append(
        {
            "油价口径": "三油+均价同时满足",
            "筛选规则": "布伦特、迪拜、WTI、三油算术均价均在[40,80] USD/bbl",
            "国际原油列": "三油算术均价_USD_per_bbl",
            **strict_stats,
        }
    )

    summary = pd.DataFrame(summary_rows)
    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return summary, samples


def make_ratio_model_svg(summary: pd.DataFrame, samples: pd.DataFrame) -> str:
    width = 1280
    height = 760
    left = 80
    right = 42
    top = 78
    bottom = 78
    gap_x = 74
    gap_y = 86
    panel_width = (width - left - right - gap_x) / 2
    panel_height = (height - top - bottom - gap_y) / 2
    colors = {
        "布伦特": "#2f6fbb",
        "迪拜": "#7a5dbb",
        "WTI": "#348f6c",
        "三油算术均价": "#222222",
    }

    if samples.empty:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<text x="20" y="30">No data</text></svg>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".fit{stroke-width:2;fill:none}",
        ".point{opacity:.72}",
        "</style>",
        f'<text class="title" x="{left}" y="30">40-80美元/桶区间：国内汽油缩放拟合国际原油</text>',
        (
            f'<text class="note" x="{left}" y="50">'
            "模型为 y = scale * x；x 为国内汽油价格(USD/bbl)，y 为对应国际原油区间均价。"
            "</text>"
        ),
    ]

    for index, (target_name, _) in enumerate(MODEL_TARGETS.items()):
        data = samples[samples["油价口径"] == target_name].copy()
        row = summary[summary["油价口径"] == target_name].iloc[0]
        panel_x = left + (index % 2) * (panel_width + gap_x)
        panel_y = top + (index // 2) * (panel_height + gap_y)

        if data.empty:
            parts.append(f'<text x="{panel_x}" y="{panel_y + 18}">{html.escape(target_name)}: No data</text>')
            continue

        x_values = data["汽油价格_USD_per_bbl"].astype(float)
        y_values = data["国际原油均价_USD_per_bbl"].astype(float)
        x_ticks = nice_ticks(float(x_values.min()), float(x_values.max()), count=5)
        y_ticks = nice_ticks(float(y_values.min()), float(y_values.max()), count=5)
        x_min, x_max = x_ticks[0], x_ticks[-1]
        y_min, y_max = y_ticks[0], y_ticks[-1]
        if math.isclose(x_min, x_max):
            x_max = x_min + 1
        if math.isclose(y_min, y_max):
            y_max = y_min + 1

        def x_scale(value: float) -> float:
            return panel_x + (value - x_min) / (x_max - x_min) * panel_width

        def y_scale(value: float) -> float:
            return panel_y + (y_max - value) / (y_max - y_min) * panel_height

        for tick in x_ticks:
            x = x_scale(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" y2="{panel_y + panel_height}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{panel_y + panel_height + 20}" text-anchor="middle">{tick:g}</text>'
            )

        for tick in y_ticks:
            y = y_scale(tick)
            parts.append(
                f'<line class="grid" x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_width}" y2="{y:.1f}"/>'
            )
            parts.append(f'<text x="{panel_x - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')

        parts.extend(
            [
                f'<line class="axis" x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}"/>',
                f'<line class="axis" x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}"/>',
                f'<text x="{panel_x}" y="{panel_y - 14}">{html.escape(target_name)} '
                f'n={int(row["样本数"])} scale={float(row["缩放系数_国际原油等于系数乘国内汽油"]):.4f}</text>',
            ]
        )

        color = colors[target_name]
        for point in data.itertuples(index=False):
            parts.append(
                f'<circle class="point" cx="{x_scale(float(point.汽油价格_USD_per_bbl)):.1f}" '
                f'cy="{y_scale(float(point.国际原油均价_USD_per_bbl)):.1f}" r="3" fill="{color}"/>'
            )

        scale_value = float(row["缩放系数_国际原油等于系数乘国内汽油"])
        fit_x_min = max(x_min, y_min / scale_value)
        fit_x_max = min(x_max, y_max / scale_value)
        if fit_x_min < fit_x_max:
            parts.append(
                f'<line class="fit" x1="{x_scale(fit_x_min):.1f}" y1="{y_scale(scale_value * fit_x_min):.1f}" '
                f'x2="{x_scale(fit_x_max):.1f}" y2="{y_scale(scale_value * fit_x_max):.1f}" stroke="{color}"/>'
            )

    parts.append(f'<text x="{left}" y="{height - 28}">横轴：国内汽油 USD/bbl；纵轴：国际原油区间均价 USD/bbl。</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_ratio_model_outputs(result_2016: pd.DataFrame) -> None:
    summary, samples = build_ratio_model_outputs(result_2016)

    sample_columns = [
        "油价口径",
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油价格_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "汽油价格_USD_per_bbl",
        "国际原油均价_USD_per_bbl",
        "国内汽油_国际原油比值",
        "缩放拟合国际原油_USD_per_bbl",
        "缩放拟合残差_USD_per_bbl",
        "布伦特均价_USD_per_bbl",
        "迪拜均价_USD_per_bbl",
        "WTI均价_USD_per_bbl",
        "三油算术均价_USD_per_bbl",
    ]
    if samples.empty:
        samples.to_csv(RATIO_MODEL_SAMPLES_2016_PATH, index=False, encoding="utf-8-sig")
    else:
        format_dates_for_output(samples[sample_columns]).sort_values(
            ["油价口径", "调整日期"],
            ascending=[True, False],
        ).to_csv(RATIO_MODEL_SAMPLES_2016_PATH, index=False, encoding="utf-8-sig")

    summary.to_csv(RATIO_MODEL_SUMMARY_2016_PATH, index=False, encoding="utf-8-sig")
    RATIO_MODEL_JSON_2016_PATH.write_text(
        json.dumps(
            {
                "date_range": {
                    "start": FILTERED_START_DATE.strftime("%Y-%m-%d"),
                    "end": END_DATE.strftime("%Y-%m-%d"),
                },
                "filter": "target international oil interval average in [40, 80] USD/bbl",
                "ratio": "国内汽油_国际原油比值 = 汽油价格_USD_per_bbl / 国际原油均价_USD_per_bbl",
                "model": "国际原油均价_USD_per_bbl = scale * 汽油价格_USD_per_bbl",
                "summary": summary.to_dict(orient="records"),
                "outputs": {
                    "samples": str(RATIO_MODEL_SAMPLES_2016_PATH),
                    "summary_csv": str(RATIO_MODEL_SUMMARY_2016_PATH),
                    "summary_json": str(RATIO_MODEL_JSON_2016_PATH),
                    "chart": str(RATIO_MODEL_CHART_2016_PATH),
                    "strict_samples": str(RATIO_MODEL_STRICT_SAMPLES_2016_PATH),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    RATIO_MODEL_CHART_2016_PATH.write_text(make_ratio_model_svg(summary, samples), encoding="utf-8")

    strict_columns = list(MODEL_TARGETS.values())
    strict = result_2016.dropna(subset=["汽油价格_USD_per_bbl", *strict_columns]).copy()
    strict = strict[
        strict[strict_columns].apply(lambda row: row.between(40, 80).all(), axis=1)
    ].copy()
    strict_columns_out = [
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油价格_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "汽油价格_USD_per_bbl",
        "布伦特均价_USD_per_bbl",
        "迪拜均价_USD_per_bbl",
        "WTI均价_USD_per_bbl",
        "三油算术均价_USD_per_bbl",
    ]
    format_dates_for_output(strict[strict_columns_out]).sort_values(
        "调整日期",
        ascending=False,
    ).to_csv(RATIO_MODEL_STRICT_SAMPLES_2016_PATH, index=False, encoding="utf-8-sig")


def fit_linear_model(sample: pd.DataFrame, target_column: str) -> dict[str, float | int | None]:
    clean = sample.dropna(subset=["汽油价格_USD_per_bbl", target_column]).copy()
    if len(clean) < 2:
        return {
            "样本数": int(len(clean)),
            "国际原油均值_USD_per_bbl": None,
            "国内汽油均值_USD_per_bbl": None,
            "截距_intercept": None,
            "斜率_slope": None,
            "线性模型RMSE_USD_per_bbl": None,
            "线性模型MAE_USD_per_bbl": None,
            "线性模型R2": None,
        }

    x = clean[target_column].astype(float)
    y = clean["汽油价格_USD_per_bbl"].astype(float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float((x_centered * x_centered).sum())
    slope = float((x_centered * y_centered).sum() / denominator) if denominator else None
    intercept = float(y.mean() - slope * x.mean()) if slope is not None else None

    if slope is None or intercept is None:
        rmse = None
        mae = None
        r2 = None
    else:
        predicted = intercept + slope * x
        residual = y - predicted
        rmse = float(math.sqrt((residual * residual).mean()))
        mae = float(residual.abs().mean())
        total = float((y_centered * y_centered).sum())
        squared_error = float((residual * residual).sum())
        r2 = float(1 - squared_error / total) if total else None

    return {
        "样本数": int(len(clean)),
        "国际原油均值_USD_per_bbl": float(x.mean()),
        "国内汽油均值_USD_per_bbl": float(y.mean()),
        "截距_intercept": intercept,
        "斜率_slope": slope,
        "线性模型RMSE_USD_per_bbl": rmse,
        "线性模型MAE_USD_per_bbl": mae,
        "线性模型R2": r2,
    }


def build_linear_model_outputs(result_2016: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []

    for target_name, target_column in MODEL_TARGETS.items():
        sample = result_2016[
            result_2016[target_column].between(40, 80, inclusive="both")
        ].copy()
        stats = fit_linear_model(sample, target_column)
        intercept = stats["截距_intercept"]
        slope = stats["斜率_slope"]
        function_text = (
            f"国内汽油 = {intercept:.6f} + {slope:.6f} * {target_name}"
            if intercept is not None and slope is not None
            else None
        )
        summary_rows.append(
            {
                "油价口径": target_name,
                "筛选规则": f"{target_name}区间均价在[40,80] USD/bbl",
                "自变量_国际原油列": target_column,
                "因变量": "汽油价格_USD_per_bbl",
                "回归函数": function_text,
                **stats,
            }
        )

        if not sample.empty and intercept is not None and slope is not None:
            sample["油价口径"] = target_name
            sample["国际原油均价_USD_per_bbl"] = sample[target_column]
            sample["国内汽油实际_USD_per_bbl"] = sample["汽油价格_USD_per_bbl"]
            sample["线性拟合国内汽油_USD_per_bbl"] = (
                float(intercept) + float(slope) * sample["国际原油均价_USD_per_bbl"]
            )
            sample["线性拟合残差_USD_per_bbl"] = (
                sample["国内汽油实际_USD_per_bbl"]
                - sample["线性拟合国内汽油_USD_per_bbl"]
            )
            sample_frames.append(sample)

    summary = pd.DataFrame(summary_rows)
    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return summary, samples


def make_linear_model_svg(summary: pd.DataFrame, samples: pd.DataFrame) -> str:
    width = 1280
    height = 760
    left = 80
    right = 42
    top = 78
    bottom = 78
    gap_x = 74
    gap_y = 86
    panel_width = (width - left - right - gap_x) / 2
    panel_height = (height - top - bottom - gap_y) / 2
    colors = {
        "布伦特": "#2f6fbb",
        "迪拜": "#7a5dbb",
        "WTI": "#348f6c",
        "三油算术均价": "#222222",
    }

    if samples.empty:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<text x="20" y="30">No data</text></svg>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".fit{stroke-width:2;fill:none}",
        ".point{opacity:.72}",
        "</style>",
        f'<text class="title" x="{left}" y="30">40-80美元/桶区间：国际原油线性拟合国内汽油</text>',
        (
            f'<text class="note" x="{left}" y="50">'
            "模型为 y = intercept + slope * x；x 为国际原油区间均价，y 为国内汽油价格，单位均为 USD/bbl。"
            "</text>"
        ),
    ]

    for index, (target_name, _) in enumerate(MODEL_TARGETS.items()):
        data = samples[samples["油价口径"] == target_name].copy()
        row = summary[summary["油价口径"] == target_name].iloc[0]
        panel_x = left + (index % 2) * (panel_width + gap_x)
        panel_y = top + (index // 2) * (panel_height + gap_y)

        if data.empty:
            parts.append(f'<text x="{panel_x}" y="{panel_y + 18}">{html.escape(target_name)}: No data</text>')
            continue

        x_values = data["国际原油均价_USD_per_bbl"].astype(float)
        y_values = data["国内汽油实际_USD_per_bbl"].astype(float)
        x_ticks = nice_ticks(float(x_values.min()), float(x_values.max()), count=5)
        y_ticks = nice_ticks(float(y_values.min()), float(y_values.max()), count=5)
        x_min, x_max = x_ticks[0], x_ticks[-1]
        y_min, y_max = y_ticks[0], y_ticks[-1]
        if math.isclose(x_min, x_max):
            x_max = x_min + 1
        if math.isclose(y_min, y_max):
            y_max = y_min + 1

        def x_scale(value: float) -> float:
            return panel_x + (value - x_min) / (x_max - x_min) * panel_width

        def y_scale(value: float) -> float:
            return panel_y + (y_max - value) / (y_max - y_min) * panel_height

        for tick in x_ticks:
            x = x_scale(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" y2="{panel_y + panel_height}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{panel_y + panel_height + 20}" text-anchor="middle">{tick:g}</text>'
            )

        for tick in y_ticks:
            y = y_scale(tick)
            parts.append(
                f'<line class="grid" x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_width}" y2="{y:.1f}"/>'
            )
            parts.append(f'<text x="{panel_x - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')

        slope = float(row["斜率_slope"])
        intercept = float(row["截距_intercept"])
        r2 = row["线性模型R2"]
        r2_text = f" R2={float(r2):.3f}" if pd.notna(r2) else ""
        parts.extend(
            [
                f'<line class="axis" x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}"/>',
                f'<line class="axis" x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}"/>',
                f'<text x="{panel_x}" y="{panel_y - 14}">{html.escape(target_name)} '
                f'n={int(row["样本数"])} y={intercept:.2f}+{slope:.3f}x{r2_text}</text>',
            ]
        )

        color = colors[target_name]
        for point in data.itertuples(index=False):
            parts.append(
                f'<circle class="point" cx="{x_scale(float(point.国际原油均价_USD_per_bbl)):.1f}" '
                f'cy="{y_scale(float(point.国内汽油实际_USD_per_bbl)):.1f}" r="3" fill="{color}"/>'
            )

        line_x_min = x_min
        line_x_max = x_max
        if slope > 0:
            line_x_min = max(line_x_min, (y_min - intercept) / slope)
            line_x_max = min(line_x_max, (y_max - intercept) / slope)
        if line_x_min < line_x_max:
            parts.append(
                f'<line class="fit" x1="{x_scale(line_x_min):.1f}" y1="{y_scale(intercept + slope * line_x_min):.1f}" '
                f'x2="{x_scale(line_x_max):.1f}" y2="{y_scale(intercept + slope * line_x_max):.1f}" stroke="{color}"/>'
            )

    parts.append(f'<text x="{left}" y="{height - 28}">横轴：国际原油区间均价 USD/bbl；纵轴：国内汽油 USD/bbl。</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_linear_model_outputs(result_2016: pd.DataFrame) -> None:
    summary, samples = build_linear_model_outputs(result_2016)

    sample_columns = [
        "油价口径",
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油价格_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "国际原油均价_USD_per_bbl",
        "国内汽油实际_USD_per_bbl",
        "线性拟合国内汽油_USD_per_bbl",
        "线性拟合残差_USD_per_bbl",
        "布伦特均价_USD_per_bbl",
        "迪拜均价_USD_per_bbl",
        "WTI均价_USD_per_bbl",
        "三油算术均价_USD_per_bbl",
    ]
    if samples.empty:
        samples.to_csv(LINEAR_MODEL_SAMPLES_2016_PATH, index=False, encoding="utf-8-sig")
    else:
        format_dates_for_output(samples[sample_columns]).sort_values(
            ["油价口径", "调整日期"],
            ascending=[True, False],
        ).to_csv(LINEAR_MODEL_SAMPLES_2016_PATH, index=False, encoding="utf-8-sig")

    summary.to_csv(LINEAR_MODEL_SUMMARY_2016_PATH, index=False, encoding="utf-8-sig")
    LINEAR_MODEL_JSON_2016_PATH.write_text(
        json.dumps(
            {
                "date_range": {
                    "start": FILTERED_START_DATE.strftime("%Y-%m-%d"),
                    "end": END_DATE.strftime("%Y-%m-%d"),
                },
                "filter": "target international oil interval average in [40, 80] USD/bbl",
                "model": "国内汽油价格_USD_per_bbl = intercept + slope * 国际原油均价_USD_per_bbl",
                "summary": summary.to_dict(orient="records"),
                "outputs": {
                    "samples": str(LINEAR_MODEL_SAMPLES_2016_PATH),
                    "summary_csv": str(LINEAR_MODEL_SUMMARY_2016_PATH),
                    "summary_json": str(LINEAR_MODEL_JSON_2016_PATH),
                    "chart": str(LINEAR_MODEL_CHART_2016_PATH),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LINEAR_MODEL_CHART_2016_PATH.write_text(
        make_linear_model_svg(summary, samples),
        encoding="utf-8",
    )


def classify_up_down_sample(
    result_2016: pd.DataFrame,
    target_name: str,
    target_column: str,
) -> pd.DataFrame:
    sample = result_2016[
        result_2016[target_column].between(40, 80, inclusive="both")
    ].copy()
    sample = sample.sort_values("调整日期").reset_index(drop=True)
    sample["油价口径"] = target_name
    sample["国际原油均价_USD_per_bbl"] = sample[target_column]
    sample["国内汽油实际_USD_per_bbl"] = sample["汽油价格_USD_per_bbl"]
    sample["上一样本国际原油均价_USD_per_bbl"] = sample[
        "国际原油均价_USD_per_bbl"
    ].shift(1)
    sample["国际原油变化_USD_per_bbl"] = (
        sample["国际原油均价_USD_per_bbl"]
        - sample["上一样本国际原油均价_USD_per_bbl"]
    )
    sample["价格方向"] = pd.NA
    sample.loc[sample["国际原油变化_USD_per_bbl"] > 0, "价格方向"] = "上升"
    sample.loc[sample["国际原油变化_USD_per_bbl"] < 0, "价格方向"] = "下降"
    return sample.dropna(subset=["价格方向"]).reset_index(drop=True)


def make_up_down_group_svg(
    target_name: str,
    summary: pd.DataFrame,
    samples: pd.DataFrame,
) -> str:
    width = 1280
    height = 520
    left = 84
    right = 44
    top = 82
    bottom = 72
    gap = 82
    panel_width = (width - left - right - gap) / 2
    panel_height = height - top - bottom
    colors = {"上升": "#c35a4a", "下降": "#2f6fbb"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".fit{stroke-width:2;fill:none}",
        ".point{opacity:.74}",
        "</style>",
        f'<text class="title" x="{left}" y="30">{html.escape(target_name)}：上升/下降区间线性回归</text>',
        (
            f'<text class="note" x="{left}" y="50">'
            "筛选国际油价40-80 USD/bbl；方向按相邻样本国际原油均价变化划分；y=国内汽油，x=国际原油。"
            "</text>"
        ),
    ]

    for index, direction in enumerate(["上升", "下降"]):
        data = samples[samples["价格方向"] == direction].copy()
        row = summary[summary["价格方向"] == direction]
        panel_x = left + index * (panel_width + gap)
        panel_y = top

        if data.empty or row.empty or pd.isna(row.iloc[0]["斜率_slope"]):
            parts.append(
                f'<text x="{panel_x}" y="{panel_y + 18}">{html.escape(direction)}: No data</text>'
            )
            continue

        row = row.iloc[0]
        x_values = data["国际原油均价_USD_per_bbl"].astype(float)
        y_values = data["国内汽油实际_USD_per_bbl"].astype(float)
        x_ticks = nice_ticks(float(x_values.min()), float(x_values.max()), count=5)
        y_ticks = nice_ticks(float(y_values.min()), float(y_values.max()), count=5)
        x_min, x_max = x_ticks[0], x_ticks[-1]
        y_min, y_max = y_ticks[0], y_ticks[-1]
        if math.isclose(x_min, x_max):
            x_max = x_min + 1
        if math.isclose(y_min, y_max):
            y_max = y_min + 1

        def x_scale(value: float) -> float:
            return panel_x + (value - x_min) / (x_max - x_min) * panel_width

        def y_scale(value: float) -> float:
            return panel_y + (y_max - value) / (y_max - y_min) * panel_height

        for tick in x_ticks:
            x = x_scale(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" y2="{panel_y + panel_height}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{panel_y + panel_height + 20}" text-anchor="middle">{tick:g}</text>'
            )

        for tick in y_ticks:
            y = y_scale(tick)
            parts.append(
                f'<line class="grid" x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_width}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{panel_x - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>'
            )

        slope = float(row["斜率_slope"])
        intercept = float(row["截距_intercept"])
        r2 = row["线性模型R2"]
        r2_text = f" R2={float(r2):.3f}" if pd.notna(r2) else ""
        parts.extend(
            [
                f'<line class="axis" x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}"/>',
                f'<line class="axis" x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}"/>',
                f'<text x="{panel_x}" y="{panel_y - 14}">{html.escape(direction)} '
                f'n={int(row["样本数"])} y={intercept:.2f}+{slope:.3f}x{r2_text}</text>',
            ]
        )

        color = colors[direction]
        for point in data.itertuples(index=False):
            parts.append(
                f'<circle class="point" cx="{x_scale(float(point.国际原油均价_USD_per_bbl)):.1f}" '
                f'cy="{y_scale(float(point.国内汽油实际_USD_per_bbl)):.1f}" r="3" fill="{color}"/>'
            )

        line_x_min = x_min
        line_x_max = x_max
        if slope > 0:
            line_x_min = max(line_x_min, (y_min - intercept) / slope)
            line_x_max = min(line_x_max, (y_max - intercept) / slope)
        if line_x_min < line_x_max:
            parts.append(
                f'<line class="fit" x1="{x_scale(line_x_min):.1f}" y1="{y_scale(intercept + slope * line_x_min):.1f}" '
                f'x2="{x_scale(line_x_max):.1f}" y2="{y_scale(intercept + slope * line_x_max):.1f}" stroke="{color}"/>'
            )

    parts.append(f'<text x="{left}" y="{height - 26}">横轴：国际原油区间均价 USD/bbl；纵轴：国内汽油 USD/bbl。</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_up_down_model_outputs(result_2016: pd.DataFrame) -> None:
    UP_DOWN_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []
    image_outputs: dict[str, str] = {}

    for target_name, target_column in MODEL_TARGETS.items():
        target_dir = UP_DOWN_ANALYZE_DIR / target_name
        target_dir.mkdir(parents=True, exist_ok=True)

        samples = classify_up_down_sample(result_2016, target_name, target_column)
        group_summary_rows: list[dict[str, object]] = []

        for direction in ["上升", "下降"]:
            subset = samples[samples["价格方向"] == direction].copy()
            stats = fit_linear_model(subset, target_column)
            intercept = stats["截距_intercept"]
            slope = stats["斜率_slope"]
            function_text = (
                f"国内汽油 = {intercept:.6f} + {slope:.6f} * {target_name}"
                if intercept is not None and slope is not None
                else None
            )
            row = {
                "油价口径": target_name,
                "价格方向": direction,
                "筛选规则": f"{target_name}区间均价在[40,80] USD/bbl，且相邻样本国际原油均价{direction}",
                "自变量_国际原油列": target_column,
                "因变量": "汽油价格_USD_per_bbl",
                "回归函数": function_text,
                **stats,
            }
            summary_rows.append(row)
            group_summary_rows.append(row)

            if not subset.empty:
                subset = subset.copy()
                if intercept is not None and slope is not None:
                    subset["线性拟合国内汽油_USD_per_bbl"] = (
                        float(intercept)
                        + float(slope) * subset["国际原油均价_USD_per_bbl"]
                    )
                    subset["线性拟合残差_USD_per_bbl"] = (
                        subset["国内汽油实际_USD_per_bbl"]
                        - subset["线性拟合国内汽油_USD_per_bbl"]
                    )
                else:
                    subset["线性拟合国内汽油_USD_per_bbl"] = pd.NA
                    subset["线性拟合残差_USD_per_bbl"] = pd.NA
                sample_frames.append(subset)

        group_summary = pd.DataFrame(group_summary_rows)
        chart_path = target_dir / "up_down_linear_model.svg"
        chart_path.write_text(
            make_up_down_group_svg(target_name, group_summary, samples),
            encoding="utf-8",
        )
        group_summary.to_csv(target_dir / "up_down_linear_model_summary.csv", index=False, encoding="utf-8-sig")
        if not samples.empty:
            sample_columns = [
                "油价口径",
                "价格方向",
                "调整日期",
                "上次调整日期",
                "汽油",
                "汽油价格_CNY_per_ton",
                "汇率日期",
                "USD_CNY",
                "国际原油均价_USD_per_bbl",
                "上一样本国际原油均价_USD_per_bbl",
                "国际原油变化_USD_per_bbl",
                "国内汽油实际_USD_per_bbl",
                "布伦特均价_USD_per_bbl",
                "迪拜均价_USD_per_bbl",
                "WTI均价_USD_per_bbl",
                "三油算术均价_USD_per_bbl",
            ]
            format_dates_for_output(samples[sample_columns]).sort_values(
                ["价格方向", "调整日期"],
                ascending=[True, False],
            ).to_csv(target_dir / "up_down_samples.csv", index=False, encoding="utf-8-sig")
        image_outputs[target_name] = str(chart_path)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(UP_DOWN_MODEL_SUMMARY_2016_PATH, index=False, encoding="utf-8-sig")

    sample_columns = [
        "油价口径",
        "价格方向",
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油价格_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "国际原油均价_USD_per_bbl",
        "上一样本国际原油均价_USD_per_bbl",
        "国际原油变化_USD_per_bbl",
        "国内汽油实际_USD_per_bbl",
        "线性拟合国内汽油_USD_per_bbl",
        "线性拟合残差_USD_per_bbl",
        "布伦特均价_USD_per_bbl",
        "迪拜均价_USD_per_bbl",
        "WTI均价_USD_per_bbl",
        "三油算术均价_USD_per_bbl",
    ]
    if sample_frames:
        all_samples = pd.concat(sample_frames, ignore_index=True)
        format_dates_for_output(all_samples[sample_columns]).sort_values(
            ["油价口径", "价格方向", "调整日期"],
            ascending=[True, True, False],
        ).to_csv(UP_DOWN_MODEL_SAMPLES_2016_PATH, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=sample_columns).to_csv(
            UP_DOWN_MODEL_SAMPLES_2016_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    UP_DOWN_MODEL_JSON_2016_PATH.write_text(
        json.dumps(
            {
                "date_range": {
                    "start": FILTERED_START_DATE.strftime("%Y-%m-%d"),
                    "end": END_DATE.strftime("%Y-%m-%d"),
                },
                "filter": "target international oil interval average in [40, 80] USD/bbl",
                "direction_rule": (
                    "sort samples by 调整日期 within each oil target; 上升 if current target "
                    "international average is greater than the previous sample, 下降 if lower"
                ),
                "model": "国内汽油价格_USD_per_bbl = intercept + slope * 国际原油均价_USD_per_bbl",
                "summary": summary.to_dict(orient="records"),
                "image_directory": str(UP_DOWN_ANALYZE_DIR),
                "image_outputs": image_outputs,
                "outputs": {
                    "summary_csv": str(UP_DOWN_MODEL_SUMMARY_2016_PATH),
                    "samples_csv": str(UP_DOWN_MODEL_SAMPLES_2016_PATH),
                    "summary_json": str(UP_DOWN_MODEL_JSON_2016_PATH),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_adjustment_change_samples_for_target(
    result: pd.DataFrame,
    target_name: str,
    target_column: str,
) -> pd.DataFrame:
    base = result.sort_values("调整日期").copy()
    base["油价口径"] = target_name
    base["当前国际均价_USD_per_bbl"] = base[target_column]
    base["上一窗口国际均价_USD_per_bbl"] = base[target_column].shift(1)
    base["国际价格变动_USD_per_bbl"] = (
        base["当前国际均价_USD_per_bbl"] - base["上一窗口国际均价_USD_per_bbl"]
    )
    base["国内调价幅度_USD_per_bbl"] = (
        base["国内调价幅度_CNY_per_ton"] / base["USD_CNY"] / BARRELS_PER_TON
    )

    if target_name == "三油算术均价":
        source_count_columns = [f"{name}样本天数" for name in OIL_SOURCES]
        current_count_valid = base[source_count_columns].apply(
            lambda row: row.between(8, 12).all(),
            axis=1,
        )
        previous_count_valid = base[source_count_columns].shift(1).apply(
            lambda row: row.between(8, 12).all(),
            axis=1,
        )
        base["当前窗口样本天数"] = base[source_count_columns].min(axis=1)
        base["上一窗口样本天数"] = base[source_count_columns].shift(1).min(axis=1)
    else:
        count_column = f"{target_name}样本天数"
        base["当前窗口样本天数"] = base[count_column]
        base["上一窗口样本天数"] = base[count_column].shift(1)
        current_count_valid = base["当前窗口样本天数"].between(8, 12, inclusive="both")
        previous_count_valid = base["上一窗口样本天数"].between(8, 12, inclusive="both")

    mask = (
        base["当前国际均价_USD_per_bbl"].between(40, 80, inclusive="both")
        & base["上一窗口国际均价_USD_per_bbl"].between(40, 80, inclusive="both")
        & current_count_valid
        & previous_count_valid
        & base["国内调价幅度_CNY_per_ton"].notna()
        & base["国内调价幅度_USD_per_bbl"].notna()
        & base["USD_CNY"].notna()
    )
    return base.loc[mask].copy()


def build_adjustment_change_samples(result: pd.DataFrame) -> pd.DataFrame:
    samples = [
        build_adjustment_change_samples_for_target(result, target_name, target_column)
        for target_name, target_column in MODEL_TARGETS.items()
    ]
    if not samples:
        return pd.DataFrame()
    return pd.concat(samples, ignore_index=True)


def make_adjustment_change_svg(samples: pd.DataFrame) -> str:
    width = 1280
    height = 760
    left = 82
    right = 46
    top = 78
    bottom = 82
    gap_x = 76
    gap_y = 92
    panel_width = (width - left - right - gap_x) / 2
    panel_height = (height - top - bottom - gap_y) / 2
    colors = {
        "布伦特": "#2f6fbb",
        "迪拜": "#7a5dbb",
        "WTI": "#348f6c",
        "三油算术均价": "#222222",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".zero{stroke:#888;stroke-width:1.2;stroke-dasharray:4 4}",
        ".point{opacity:.76}",
        "</style>",
        f'<text class="title" x="{left}" y="30">国内调价幅度 vs 国际价格变动</text>',
        (
            f'<text class="note" x="{left}" y="50">'
            "筛选当前/上一国际均价均在40-80 USD/bbl，且调价窗口有效样本天数在[8,12]。"
            "</text>"
        ),
    ]

    for index, (target_name, _) in enumerate(MODEL_TARGETS.items()):
        data = samples[samples["油价口径"] == target_name].copy()
        panel_x = left + (index % 2) * (panel_width + gap_x)
        panel_y = top + (index // 2) * (panel_height + gap_y)

        if data.empty:
            parts.append(
                f'<text x="{panel_x}" y="{panel_y + 18}">{html.escape(target_name)} n=0</text>'
            )
            continue

        x_values = data["国际价格变动_USD_per_bbl"].astype(float)
        y_values = data["国内调价幅度_USD_per_bbl"].astype(float)
        x_ticks = nice_ticks(min(float(x_values.min()), 0.0), max(float(x_values.max()), 0.0), count=5)
        y_ticks = nice_ticks(min(float(y_values.min()), 0.0), max(float(y_values.max()), 0.0), count=5)
        x_min, x_max = x_ticks[0], x_ticks[-1]
        y_min, y_max = y_ticks[0], y_ticks[-1]
        if math.isclose(x_min, x_max):
            x_max = x_min + 1
        if math.isclose(y_min, y_max):
            y_max = y_min + 1

        def x_scale(value: float) -> float:
            return panel_x + (value - x_min) / (x_max - x_min) * panel_width

        def y_scale(value: float) -> float:
            return panel_y + (y_max - value) / (y_max - y_min) * panel_height

        for tick in x_ticks:
            x = x_scale(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" y2="{panel_y + panel_height}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{panel_y + panel_height + 20}" text-anchor="middle">{tick:g}</text>'
            )

        for tick in y_ticks:
            y = y_scale(tick)
            parts.append(
                f'<line class="grid" x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_width}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{panel_x - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>'
            )

        if x_min <= 0 <= x_max:
            x0 = x_scale(0)
            parts.append(
                f'<line class="zero" x1="{x0:.1f}" y1="{panel_y}" x2="{x0:.1f}" y2="{panel_y + panel_height}"/>'
            )
        if y_min <= 0 <= y_max:
            y0 = y_scale(0)
            parts.append(
                f'<line class="zero" x1="{panel_x}" y1="{y0:.1f}" x2="{panel_x + panel_width}" y2="{y0:.1f}"/>'
            )

        parts.extend(
            [
                f'<line class="axis" x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}"/>',
                f'<line class="axis" x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}"/>',
                f'<text x="{panel_x}" y="{panel_y - 14}">{html.escape(target_name)} n={len(data)}</text>',
            ]
        )

        color = colors[target_name]
        for point in data.itertuples(index=False):
            parts.append(
                f'<circle class="point" cx="{x_scale(float(point.国际价格变动_USD_per_bbl)):.1f}" '
                f'cy="{y_scale(float(point.国内调价幅度_USD_per_bbl)):.1f}" r="3" fill="{color}"/>'
            )

    parts.append(
        f'<text x="{left}" y="{height - 34}">横轴：国际价格变动（当前窗口均价 - 上一窗口均价，USD/bbl）；纵轴：国内调价幅度（USD/bbl）。</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def fit_adjustment_change_linear_model(sample: pd.DataFrame) -> dict[str, float | int | None]:
    clean = sample.dropna(
        subset=["国际价格变动_USD_per_bbl", "国内调价幅度_USD_per_bbl"]
    ).copy()
    if len(clean) < 2:
        return {
            "样本数": int(len(clean)),
            "国际价格变动均值_USD_per_bbl": None,
            "国内调价幅度均值_USD_per_bbl": None,
            "截距_intercept": None,
            "斜率_slope": None,
            "线性模型RMSE_USD_per_bbl": None,
            "线性模型MAE_USD_per_bbl": None,
            "线性模型R2": None,
        }

    x = clean["国际价格变动_USD_per_bbl"].astype(float)
    y = clean["国内调价幅度_USD_per_bbl"].astype(float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float((x_centered * x_centered).sum())
    slope = float((x_centered * y_centered).sum() / denominator) if denominator else None
    intercept = float(y.mean() - slope * x.mean()) if slope is not None else None

    if slope is None or intercept is None:
        rmse = None
        mae = None
        r2 = None
    else:
        predicted = intercept + slope * x
        residual = y - predicted
        rmse = float(math.sqrt((residual * residual).mean()))
        mae = float(residual.abs().mean())
        total = float((y_centered * y_centered).sum())
        squared_error = float((residual * residual).sum())
        r2 = float(1 - squared_error / total) if total else None

    return {
        "样本数": int(len(clean)),
        "国际价格变动均值_USD_per_bbl": float(x.mean()),
        "国内调价幅度均值_USD_per_bbl": float(y.mean()),
        "截距_intercept": intercept,
        "斜率_slope": slope,
        "线性模型RMSE_USD_per_bbl": rmse,
        "线性模型MAE_USD_per_bbl": mae,
        "线性模型R2": r2,
    }


def build_adjustment_change_linear_outputs(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = samples[samples["调整日期"] >= FILTERED_START_DATE].copy()
    summary_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []

    for target_name in MODEL_TARGETS:
        target_samples = filtered[filtered["油价口径"] == target_name].copy()
        stats = fit_adjustment_change_linear_model(target_samples)
        intercept = stats["截距_intercept"]
        slope = stats["斜率_slope"]
        function_text = (
            f"国内调价幅度 = {intercept:.6f} + {slope:.6f} * 国际价格变动"
            if intercept is not None and slope is not None
            else None
        )
        summary_rows.append(
            {
                "油价口径": target_name,
                "筛选规则": (
                    "调整日期>=2016-01-01，当前/上一国际均价均在[40,80] USD/bbl，"
                    "当前/上一窗口样本天数均在[8,12]"
                ),
                "自变量": "国际价格变动_USD_per_bbl",
                "因变量": "国内调价幅度_USD_per_bbl",
                "回归函数": function_text,
                **stats,
            }
        )

        if not target_samples.empty:
            target_samples = target_samples.copy()
            if intercept is not None and slope is not None:
                target_samples["线性拟合国内调价幅度_USD_per_bbl"] = (
                    float(intercept)
                    + float(slope) * target_samples["国际价格变动_USD_per_bbl"]
                )
                target_samples["线性拟合残差_USD_per_bbl"] = (
                    target_samples["国内调价幅度_USD_per_bbl"]
                    - target_samples["线性拟合国内调价幅度_USD_per_bbl"]
                )
            else:
                target_samples["线性拟合国内调价幅度_USD_per_bbl"] = pd.NA
                target_samples["线性拟合残差_USD_per_bbl"] = pd.NA
            sample_frames.append(target_samples)

    summary = pd.DataFrame(summary_rows)
    model_samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return summary, model_samples


def make_adjustment_change_linear_svg(summary: pd.DataFrame, samples: pd.DataFrame) -> str:
    width = 1280
    height = 760
    left = 82
    right = 46
    top = 78
    bottom = 82
    gap_x = 76
    gap_y = 92
    panel_width = (width - left - right - gap_x) / 2
    panel_height = (height - top - bottom - gap_y) / 2
    colors = {
        "布伦特": "#2f6fbb",
        "迪拜": "#7a5dbb",
        "WTI": "#348f6c",
        "三油算术均价": "#222222",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".zero{stroke:#888;stroke-width:1.2;stroke-dasharray:4 4}",
        ".fit{stroke-width:2;fill:none}",
        ".point{opacity:.76}",
        "</style>",
        f'<text class="title" x="{left}" y="30">2016年以来：国内调价幅度线性拟合国际价格变动</text>',
        (
            f'<text class="note" x="{left}" y="50">'
            "模型：国内调价幅度 = intercept + slope * 国际价格变动；单位均为 USD/bbl。"
            "</text>"
        ),
    ]

    for index, (target_name, _) in enumerate(MODEL_TARGETS.items()):
        data = samples[samples["油价口径"] == target_name].copy()
        row = summary[summary["油价口径"] == target_name]
        panel_x = left + (index % 2) * (panel_width + gap_x)
        panel_y = top + (index // 2) * (panel_height + gap_y)

        if data.empty or row.empty or pd.isna(row.iloc[0]["斜率_slope"]):
            parts.append(
                f'<text x="{panel_x}" y="{panel_y + 18}">{html.escape(target_name)} n=0</text>'
            )
            continue

        row = row.iloc[0]
        x_values = data["国际价格变动_USD_per_bbl"].astype(float)
        y_values = data["国内调价幅度_USD_per_bbl"].astype(float)
        x_ticks = nice_ticks(min(float(x_values.min()), 0.0), max(float(x_values.max()), 0.0), count=5)
        y_ticks = nice_ticks(min(float(y_values.min()), 0.0), max(float(y_values.max()), 0.0), count=5)
        x_min, x_max = x_ticks[0], x_ticks[-1]
        y_min, y_max = y_ticks[0], y_ticks[-1]
        if math.isclose(x_min, x_max):
            x_max = x_min + 1
        if math.isclose(y_min, y_max):
            y_max = y_min + 1

        def x_scale(value: float) -> float:
            return panel_x + (value - x_min) / (x_max - x_min) * panel_width

        def y_scale(value: float) -> float:
            return panel_y + (y_max - value) / (y_max - y_min) * panel_height

        for tick in x_ticks:
            x = x_scale(tick)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{panel_y}" x2="{x:.1f}" y2="{panel_y + panel_height}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{panel_y + panel_height + 20}" text-anchor="middle">{tick:g}</text>'
            )

        for tick in y_ticks:
            y = y_scale(tick)
            parts.append(
                f'<line class="grid" x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_width}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{panel_x - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>'
            )

        if x_min <= 0 <= x_max:
            x0 = x_scale(0)
            parts.append(
                f'<line class="zero" x1="{x0:.1f}" y1="{panel_y}" x2="{x0:.1f}" y2="{panel_y + panel_height}"/>'
            )
        if y_min <= 0 <= y_max:
            y0 = y_scale(0)
            parts.append(
                f'<line class="zero" x1="{panel_x}" y1="{y0:.1f}" x2="{panel_x + panel_width}" y2="{y0:.1f}"/>'
            )

        slope = float(row["斜率_slope"])
        intercept = float(row["截距_intercept"])
        r2 = row["线性模型R2"]
        r2_text = f" R2={float(r2):.3f}" if pd.notna(r2) else ""
        parts.extend(
            [
                f'<line class="axis" x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}"/>',
                f'<line class="axis" x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}"/>',
                f'<text x="{panel_x}" y="{panel_y - 14}">{html.escape(target_name)} '
                f'n={int(row["样本数"])} y={intercept:.2f}+{slope:.3f}x{r2_text}</text>',
            ]
        )

        color = colors[target_name]
        for point in data.itertuples(index=False):
            parts.append(
                f'<circle class="point" cx="{x_scale(float(point.国际价格变动_USD_per_bbl)):.1f}" '
                f'cy="{y_scale(float(point.国内调价幅度_USD_per_bbl)):.1f}" r="3" fill="{color}"/>'
            )

        line_x_min = x_min
        line_x_max = x_max
        if not math.isclose(slope, 0):
            low_cross = (y_min - intercept) / slope
            high_cross = (y_max - intercept) / slope
            line_x_min = max(line_x_min, min(low_cross, high_cross))
            line_x_max = min(line_x_max, max(low_cross, high_cross))
        if line_x_min < line_x_max:
            parts.append(
                f'<line class="fit" x1="{x_scale(line_x_min):.1f}" y1="{y_scale(intercept + slope * line_x_min):.1f}" '
                f'x2="{x_scale(line_x_max):.1f}" y2="{y_scale(intercept + slope * line_x_max):.1f}" stroke="{color}"/>'
            )

    parts.append(
        f'<text x="{left}" y="{height - 34}">横轴：国际价格变动 USD/bbl；纵轴：国内调价幅度 USD/bbl；样本日期 >= 2016-01-01。</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_adjustment_change_linear_outputs(samples: pd.DataFrame) -> None:
    summary, model_samples = build_adjustment_change_linear_outputs(samples)
    output_columns = [
        "油价口径",
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油涨跌",
        "国内调价幅度_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "汇率间隔天数",
        "国内调价幅度_USD_per_bbl",
        "当前国际均价_USD_per_bbl",
        "上一窗口国际均价_USD_per_bbl",
        "国际价格变动_USD_per_bbl",
        "线性拟合国内调价幅度_USD_per_bbl",
        "线性拟合残差_USD_per_bbl",
        "当前窗口样本天数",
        "上一窗口样本天数",
        "布伦特均价_USD_per_bbl",
        "布伦特样本天数",
        "迪拜均价_USD_per_bbl",
        "迪拜样本天数",
        "WTI均价_USD_per_bbl",
        "WTI样本天数",
        "三油算术均价_USD_per_bbl",
        "三油可用数量",
    ]
    if model_samples.empty:
        pd.DataFrame(columns=output_columns).to_csv(
            ADJUSTMENT_CHANGE_LINEAR_SAMPLES_2016_PATH,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        format_dates_for_output(model_samples[output_columns]).sort_values(
            ["油价口径", "调整日期"],
            ascending=[True, False],
        ).to_csv(
            ADJUSTMENT_CHANGE_LINEAR_SAMPLES_2016_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    summary.to_csv(
        ADJUSTMENT_CHANGE_LINEAR_SUMMARY_2016_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    ADJUSTMENT_CHANGE_LINEAR_CHART_2016_PATH.write_text(
        make_adjustment_change_linear_svg(summary, model_samples),
        encoding="utf-8",
    )

    ADJUSTMENT_CHANGE_LINEAR_JSON_2016_PATH.write_text(
        json.dumps(
            {
                "date_range": {
                    "start": FILTERED_START_DATE.strftime("%Y-%m-%d"),
                    "end": model_samples["调整日期"].max().strftime("%Y-%m-%d")
                    if not model_samples.empty
                    else None,
                },
                "filter": (
                    "adjustment date >= 2016-01-01; current and previous international "
                    "window averages in [40, 80] USD/bbl; current and previous sample "
                    "counts in [8, 12]"
                ),
                "model": (
                    "国内调价幅度_USD_per_bbl = intercept + slope * "
                    "国际价格变动_USD_per_bbl"
                ),
                "summary": summary.to_dict(orient="records"),
                "outputs": {
                    "samples": str(ADJUSTMENT_CHANGE_LINEAR_SAMPLES_2016_PATH),
                    "summary_csv": str(ADJUSTMENT_CHANGE_LINEAR_SUMMARY_2016_PATH),
                    "summary_json": str(ADJUSTMENT_CHANGE_LINEAR_JSON_2016_PATH),
                    "chart": str(ADJUSTMENT_CHANGE_LINEAR_CHART_2016_PATH),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_adjustment_change_outputs(result: pd.DataFrame) -> None:
    OUTPUT3_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_adjustment_change_samples(result)

    output_columns = [
        "油价口径",
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油涨跌",
        "国内调价幅度_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "汇率间隔天数",
        "国内调价幅度_USD_per_bbl",
        "当前国际均价_USD_per_bbl",
        "上一窗口国际均价_USD_per_bbl",
        "国际价格变动_USD_per_bbl",
        "当前窗口样本天数",
        "上一窗口样本天数",
        "布伦特均价_USD_per_bbl",
        "布伦特样本天数",
        "迪拜均价_USD_per_bbl",
        "迪拜样本天数",
        "WTI均价_USD_per_bbl",
        "WTI样本天数",
        "三油算术均价_USD_per_bbl",
        "三油可用数量",
    ]
    if samples.empty:
        pd.DataFrame(columns=output_columns).to_csv(
            ADJUSTMENT_CHANGE_SAMPLES_PATH,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        format_dates_for_output(samples[output_columns]).sort_values(
            ["油价口径", "调整日期"],
            ascending=[True, False],
        ).to_csv(ADJUSTMENT_CHANGE_SAMPLES_PATH, index=False, encoding="utf-8-sig")

    ADJUSTMENT_CHANGE_CHART_PATH.write_text(
        make_adjustment_change_svg(samples),
        encoding="utf-8",
    )
    write_adjustment_change_linear_outputs(samples)

    sample_count_by_target = {
        target_name: int((samples["油价口径"] == target_name).sum()) if not samples.empty else 0
        for target_name in MODEL_TARGETS
    }
    summary = {
        "date_range": {
            "start": samples["调整日期"].min().strftime("%Y-%m-%d") if not samples.empty else None,
            "end": samples["调整日期"].max().strftime("%Y-%m-%d") if not samples.empty else None,
            "source_start": result["调整日期"].min().strftime("%Y-%m-%d") if not result.empty else None,
            "source_end": result["调整日期"].max().strftime("%Y-%m-%d") if not result.empty else None,
        },
        "window_rule": "current row uses oil-price observations where prev_date < date <= current_date",
        "international_change_definition": (
            "国际价格变动_USD_per_bbl = 当前调价窗口国际均价 - 上一调价窗口国际均价"
        ),
        "domestic_adjustment_definition": (
            "国内调价幅度_CNY_per_ton = product_oil.csv 中 汽油涨跌 解析值"
        ),
        "domestic_adjustment_conversion": (
            "国内调价幅度_USD_per_bbl = 国内调价幅度_CNY_per_ton / USD_CNY / 7.33"
        ),
        "fx_alignment": (
            "merge_asof backward: use nearest previous USD_CNY close within "
            f"{FX_MAX_STALE_DAYS} days"
        ),
        "oil_price_filter": (
            "current and previous international window averages must both be in [40, 80] USD/bbl; "
            "current and previous window sample counts must both be in [8, 12]"
        ),
        "sample_count_by_target": sample_count_by_target,
        "outputs": {
            "samples": str(ADJUSTMENT_CHANGE_SAMPLES_PATH),
            "summary": str(ADJUSTMENT_CHANGE_SUMMARY_PATH),
            "chart": str(ADJUSTMENT_CHANGE_CHART_PATH),
            "linear_samples_from_2016": str(ADJUSTMENT_CHANGE_LINEAR_SAMPLES_2016_PATH),
            "linear_summary_from_2016": str(ADJUSTMENT_CHANGE_LINEAR_SUMMARY_2016_PATH),
            "linear_summary_json_from_2016": str(ADJUSTMENT_CHANGE_LINEAR_JSON_2016_PATH),
            "linear_chart_from_2016": str(ADJUSTMENT_CHANGE_LINEAR_CHART_2016_PATH),
        },
    }
    ADJUSTMENT_CHANGE_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_result_files(
    result: pd.DataFrame,
    oil_prices: dict[str, pd.DataFrame],
    table_path: Path,
    chart_path: Path,
    summary_path: Path,
    start_date: pd.Timestamp,
) -> None:
    output_columns = [
        "调整日期",
        "上次调整日期",
        "汽油",
        "汽油价格_CNY_per_ton",
        "汇率日期",
        "USD_CNY",
        "汇率间隔天数",
        "汽油价格_USD_per_bbl",
        "布伦特均价_USD_per_bbl",
        "布伦特样本天数",
        "迪拜均价_USD_per_bbl",
        "迪拜样本天数",
        "WTI均价_USD_per_bbl",
        "WTI样本天数",
        "三油算术均价_USD_per_bbl",
        "三油可用数量",
    ]
    output = format_dates_for_output(result[output_columns])
    output.sort_values("调整日期", ascending=False).to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

    chart_path.write_text(make_line_svg(result), encoding="utf-8")

    summary = {
        "date_reference": str(PRODUCT_OIL_PATH),
        "date_range": {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": END_DATE.strftime("%Y-%m-%d"),
        },
        "interval_rule": "current row uses oil-price observations where prev_date < date <= current_date",
        "product_price_conversion": (
            "gasoline_USD_per_bbl = gasoline_CNY_per_ton / USD_CNY / 7.33"
        ),
        "barrels_per_ton": BARRELS_PER_TON,
        "fx_alignment": (
            "merge_asof backward: use nearest previous USD_CNY close within "
            f"{FX_MAX_STALE_DAYS} days"
        ),
        "product_rows": int(len(result)),
        "first_adjustment_date": result["调整日期"].min().strftime("%Y-%m-%d")
        if not result.empty
        else None,
        "last_adjustment_date": result["调整日期"].max().strftime("%Y-%m-%d")
        if not result.empty
        else None,
        "oil_source_rows": {name: int(len(prices)) for name, prices in oil_prices.items()},
        "outputs": {
            "table": str(table_path),
            "chart": str(chart_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(result: pd.DataFrame, oil_prices: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_result_files(
        result,
        oil_prices,
        OUTPUT_TABLE_PATH,
        OUTPUT_CHART_PATH,
        SUMMARY_PATH,
        START_DATE,
    )

    result_2016 = result[result["调整日期"] >= FILTERED_START_DATE].copy()
    write_result_files(
        result_2016,
        oil_prices,
        OUTPUT_TABLE_2016_PATH,
        OUTPUT_CHART_2016_PATH,
        SUMMARY_2016_PATH,
        FILTERED_START_DATE,
    )
    write_ratio_model_outputs(result_2016)
    write_linear_model_outputs(result_2016)
    write_up_down_model_outputs(result_2016)
    write_adjustment_change_outputs(result)


def main() -> None:
    product = read_product_oil()
    fx = read_usd_cny()
    product_with_fx = add_product_usd_per_bbl(product, fx)
    oil_prices = {name: read_oil_price(path) for name, path in OIL_SOURCES.items()}
    result = add_interval_averages(product_with_fx, oil_prices)
    write_outputs(result, oil_prices)

    print(f"Rows: {len(result)}")
    print(f"Saved table: {OUTPUT_TABLE_PATH}")
    print(f"Saved chart: {OUTPUT_CHART_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Rows from 2016: {(result['调整日期'] >= FILTERED_START_DATE).sum()}")
    print(f"Saved 2016 table: {OUTPUT_TABLE_2016_PATH}")
    print(f"Saved 2016 chart: {OUTPUT_CHART_2016_PATH}")
    print(f"Saved 2016 summary: {SUMMARY_2016_PATH}")
    print(f"Saved 40-80 model samples: {RATIO_MODEL_SAMPLES_2016_PATH}")
    print(f"Saved 40-80 model summary: {RATIO_MODEL_SUMMARY_2016_PATH}")
    print(f"Saved 40-80 model chart: {RATIO_MODEL_CHART_2016_PATH}")
    print(f"Saved 40-80 linear samples: {LINEAR_MODEL_SAMPLES_2016_PATH}")
    print(f"Saved 40-80 linear summary: {LINEAR_MODEL_SUMMARY_2016_PATH}")
    print(f"Saved 40-80 linear chart: {LINEAR_MODEL_CHART_2016_PATH}")
    print(f"Saved up/down image directory: {UP_DOWN_ANALYZE_DIR}")
    print(f"Saved up/down summary: {UP_DOWN_MODEL_SUMMARY_2016_PATH}")
    print(f"Saved output3 samples: {ADJUSTMENT_CHANGE_SAMPLES_PATH}")
    print(f"Saved output3 chart: {ADJUSTMENT_CHANGE_CHART_PATH}")
    print(f"Saved output3 2016 linear summary: {ADJUSTMENT_CHANGE_LINEAR_SUMMARY_2016_PATH}")
    print(f"Saved output3 2016 linear chart: {ADJUSTMENT_CHANGE_LINEAR_CHART_2016_PATH}")


if __name__ == "__main__":
    main()
