from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "datas" / "output"

INPUT_PATH = OUTPUT_DIR / "gasoline_basket_40_80_all_aligned.csv"

ASYMMETRY_SAMPLES_PATH = OUTPUT_DIR / "task1_asymmetry_samples.csv"
ASYMMETRY_SUMMARY_PATH = OUTPUT_DIR / "task1_asymmetry_summary.csv"
ASYMMETRY_SVG_PATH = OUTPUT_DIR / "task1_asymmetry_transmission.svg"

INTERVAL_SUMMARY_PATH = OUTPUT_DIR / "task1_oil_interval_summary.csv"
INTERVAL_SAMPLES_PATH = OUTPUT_DIR / "task1_oil_interval_samples.csv"
INTERVAL_SVG_PATH = OUTPUT_DIR / "task1_oil_interval_transmission.svg"

THRESHOLD_TABLE_PATH = OUTPUT_DIR / "task1_threshold_50_check.csv"
THRESHOLD_SUMMARY_PATH = OUTPUT_DIR / "task1_threshold_50_summary.csv"

REPORT_PATH = OUTPUT_DIR / "task1_supplement_report.json"

START_DATE = pd.Timestamp("2016-01-01")
THEORETICAL_CHANGE = "理论国内成本_CNY_per_ton_涨跌值"
ACTUAL_CHANGE = "汽油涨跌真实值_CNY_per_ton"
CURRENT_USD = "理论国内成本_USD_per_bbl_本期10工作日均值"
PREVIOUS_USD = "理论国内成本_USD_per_bbl_上期10工作日均值"
TRANSMISSION = "传递系数_实际调幅除以理论调幅"
MIN_DENOMINATOR = 1e-9
THRESHOLD_CNY_PER_TON = 50.0


def read_base_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["调整日期"] = pd.to_datetime(df["调整日期"], errors="coerce")
    numeric_columns = [
        "汽油价格真实值_CNY_per_ton",
        ACTUAL_CHANGE,
        "计算日汇率_USD_CNY",
        CURRENT_USD,
        PREVIOUS_USD,
        "理论国内成本_CNY_per_ton_本期10工作日均值",
        "理论国内成本_CNY_per_ton_上期10工作日均值",
        THEORETICAL_CHANGE,
        "汽油价格变化率",
        "理论国内成本_CNY_per_ton_变化率",
        "汽油变化率_理论国内成本CNY变化率比值",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["调整日期"] >= START_DATE].copy()
    df = df.dropna(subset=["调整日期", CURRENT_USD, PREVIOUS_USD, THEORETICAL_CHANGE])
    df = df.sort_values("调整日期").reset_index(drop=True)
    df["油价区间"] = df[CURRENT_USD].map(classify_oil_interval)
    df["上期油价区间"] = df[PREVIOUS_USD].map(classify_oil_interval)
    df["理论调幅方向"] = np.where(df[THEORETICAL_CHANGE] > 0, "上涨", "下跌")
    df.loc[df[THEORETICAL_CHANGE].abs() <= MIN_DENOMINATOR, "理论调幅方向"] = "持平"
    df[TRANSMISSION] = df[ACTUAL_CHANGE] / df[THEORETICAL_CHANGE]
    df["同向调价"] = np.sign(df[ACTUAL_CHANGE]) == np.sign(df[THEORETICAL_CHANGE])
    return df


def classify_oil_interval(value: float) -> str:
    if pd.isna(value):
        return "缺失"
    if value < 40:
        return "<40"
    if value <= 80:
        return "40-80"
    if value < 130:
        return "80-130"
    return ">=130"


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def welch_t_normal_approx(a: pd.Series, b: pd.Series) -> dict[str, float | None]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "df": None, "p_value_normal_approx": None}

    mean_a = float(a.mean())
    mean_b = float(b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    denom = math.sqrt(var_a / len(a) + var_b / len(b))
    if denom <= MIN_DENOMINATOR:
        return {"t_stat": None, "df": None, "p_value_normal_approx": None}

    t_stat = (mean_a - mean_b) / denom
    df_num = (var_a / len(a) + var_b / len(b)) ** 2
    df_den = (var_a**2) / (len(a) ** 2 * (len(a) - 1)) + (var_b**2) / (
        len(b) ** 2 * (len(b) - 1)
    )
    df = df_num / df_den if df_den > 0 else None
    p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    return {"t_stat": float(t_stat), "df": float(df) if df is not None else None, "p_value_normal_approx": float(p_value)}


def summarize_group(samples: pd.DataFrame, group_name: str) -> dict[str, object]:
    transmission = samples[TRANSMISSION].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "分组": group_name,
        "样本数": int(len(samples)),
        "有效传递系数样本数": int(len(transmission)),
        "平均理论调幅_CNY_per_ton": float(samples[THEORETICAL_CHANGE].mean()) if len(samples) else None,
        "平均实际调幅_CNY_per_ton": float(samples[ACTUAL_CHANGE].mean()) if len(samples) else None,
        "平均传递系数": float(transmission.mean()) if len(transmission) else None,
        "中位数传递系数": float(transmission.median()) if len(transmission) else None,
        "同向调价比例": float(samples["同向调价"].mean()) if len(samples) else None,
    }


def summarize_asymmetry_pair(samples: pd.DataFrame, scope: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    up = samples[samples["样本类型"] == "上涨样本"]
    down = samples[samples["样本类型"] == "下跌样本"]
    up_summary = summarize_group(up, "上涨样本")
    down_summary = summarize_group(down, "下跌样本")
    up_summary["口径"] = scope
    down_summary["口径"] = scope

    mean_up = up_summary["平均传递系数"]
    mean_down = down_summary["平均传递系数"]
    asymmetry_index = (
        float(mean_up - mean_down)
        if mean_up is not None and mean_down is not None
        else None
    )
    test = welch_t_normal_approx(up[TRANSMISSION], down[TRANSMISSION])
    comparison = {
        "scope": scope,
        "up_count": int(len(up)),
        "down_count": int(len(down)),
        "mean_up": float(mean_up) if mean_up is not None else None,
        "mean_down": float(mean_down) if mean_down is not None else None,
        "median_up": float(up[TRANSMISSION].median()) if len(up) else None,
        "median_down": float(down[TRANSMISSION].median()) if len(down) else None,
        "asymmetry_index": asymmetry_index,
        "welch_test_normal_approx": test,
        "conclusion": (
            "存在涨多跌少的非对称性"
            if asymmetry_index is not None and asymmetry_index > 0
            else "未观察到上涨传递更充分的非对称性"
        ),
    }
    return [up_summary, down_summary], comparison


def run_asymmetry_analysis(df: pd.DataFrame) -> dict[str, object]:
    samples = df.dropna(subset=[ACTUAL_CHANGE, THEORETICAL_CHANGE, TRANSMISSION]).copy()
    samples = samples[samples[THEORETICAL_CHANGE].abs() > MIN_DENOMINATOR]
    samples = samples.replace([np.inf, -np.inf], np.nan).dropna(subset=[TRANSMISSION])
    samples["样本类型"] = np.where(samples[THEORETICAL_CHANGE] > 0, "上涨样本", "下跌样本")
    samples["是否超过50元门槛"] = samples[THEORETICAL_CHANGE].abs() >= THRESHOLD_CNY_PER_TON

    raw_rows, raw_comparison = summarize_asymmetry_pair(samples, "原始口径：全部可计算窗口")
    effective_samples = samples[samples["是否超过50元门槛"]].copy()
    effective_rows, effective_comparison = summarize_asymmetry_pair(
        effective_samples,
        "主口径：剔除|理论调幅|<50元窗口",
    )
    summary = pd.DataFrame(raw_rows + effective_rows)
    summary = summary[
        [
            "口径",
            "分组",
            "样本数",
            "有效传递系数样本数",
            "平均理论调幅_CNY_per_ton",
            "平均实际调幅_CNY_per_ton",
            "平均传递系数",
            "中位数传递系数",
            "同向调价比例",
        ]
    ]
    summary["非对称指数_上涨均值减下跌均值"] = None
    summary["Welch_t统计量_正态近似"] = None
    summary["Welch_df"] = None
    summary["p值_正态近似"] = None
    for comparison in [raw_comparison, effective_comparison]:
        mask = summary["口径"] == comparison["scope"]
        summary.loc[mask, "非对称指数_上涨均值减下跌均值"] = comparison["asymmetry_index"]
        summary.loc[mask, "Welch_t统计量_正态近似"] = comparison["welch_test_normal_approx"]["t_stat"]
        summary.loc[mask, "Welch_df"] = comparison["welch_test_normal_approx"]["df"]
        summary.loc[mask, "p值_正态近似"] = comparison["welch_test_normal_approx"]["p_value_normal_approx"]

    samples = samples.sort_values("调整日期", ascending=False)
    samples["调整日期"] = samples["调整日期"].dt.strftime("%Y-%m-%d")
    samples.to_csv(ASYMMETRY_SAMPLES_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(ASYMMETRY_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    make_two_bar_svg(
        summary[summary["口径"] == "主口径：剔除|理论调幅|<50元窗口"],
        "分组",
        "平均传递系数",
        ASYMMETRY_SVG_PATH,
        "上涨/下跌窗口传递系数对比（剔除<50元理论调幅）",
        "传递系数 = 汽油真实调幅 / 理论成本调幅（CNY/ton）；主图采用50元门槛后的有效窗口。",
    )

    return {
        "raw": raw_comparison,
        "effective_abs_theoretical_change_ge_50": effective_comparison,
        "preferred_conclusion": effective_comparison["conclusion"],
        "method_note": "原始口径易受|理论调幅|很小的分母影响；主口径按机制50元门槛剔除小理论调幅窗口。",
    }


def run_interval_analysis(df: pd.DataFrame) -> dict[str, object]:
    samples = df.dropna(subset=[ACTUAL_CHANGE, THEORETICAL_CHANGE, TRANSMISSION]).copy()
    samples = samples.replace([np.inf, -np.inf], np.nan).dropna(subset=[TRANSMISSION])
    interval_order = ["<40", "40-80", "80-130", ">=130"]
    rows = []
    for interval in interval_order:
        part = samples[samples["油价区间"] == interval]
        rows.append(
            {
                "油价区间": interval,
                "样本数": int(len(part)),
                "本期均值_USD_per_bbl_最小值": float(part[CURRENT_USD].min()) if len(part) else None,
                "本期均值_USD_per_bbl_最大值": float(part[CURRENT_USD].max()) if len(part) else None,
                "平均理论调幅_CNY_per_ton": float(part[THEORETICAL_CHANGE].mean()) if len(part) else None,
                "平均实际调幅_CNY_per_ton": float(part[ACTUAL_CHANGE].mean()) if len(part) else None,
                "平均传递系数": float(part[TRANSMISSION].mean()) if len(part) else None,
                "中位数传递系数": float(part[TRANSMISSION].median()) if len(part) else None,
                "同向调价比例": float(part["同向调价"].mean()) if len(part) else None,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(INTERVAL_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    sample_export = samples.sort_values("调整日期", ascending=False).copy()
    sample_export["调整日期"] = sample_export["调整日期"].dt.strftime("%Y-%m-%d")
    sample_export.to_csv(INTERVAL_SAMPLES_PATH, index=False, encoding="utf-8-sig")

    make_interval_svg(summary, INTERVAL_SVG_PATH)
    return {
        "rows": rows,
        "sample_count": int(len(samples)),
    }


def run_threshold_analysis(df: pd.DataFrame) -> dict[str, object]:
    samples = df.dropna(subset=[THEORETICAL_CHANGE]).copy()
    samples["理论调幅绝对值_CNY_per_ton"] = samples[THEORETICAL_CHANGE].abs()
    threshold = samples[samples["理论调幅绝对值_CNY_per_ton"] < THRESHOLD_CNY_PER_TON].copy()
    threshold["实际是否未调"] = threshold[ACTUAL_CHANGE].fillna(0).abs() <= MIN_DENOMINATOR
    threshold["实际是否有调价"] = ~threshold["实际是否未调"]
    threshold["门槛规则观察判断"] = np.where(
        threshold["实际是否未调"],
        "符合：理论调幅<50且实际未调",
        "不符合或受累积机制影响：理论调幅<50但实际有调价",
    )

    export = threshold.sort_values("调整日期", ascending=False).copy()
    export["调整日期"] = export["调整日期"].dt.strftime("%Y-%m-%d")
    export.to_csv(THRESHOLD_TABLE_PATH, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "检查口径": "2016年后、已有理论窗口的调价日期",
                "理论调幅绝对值小于50样本数": int(len(threshold)),
                "其中实际未调样本数": int(threshold["实际是否未调"].sum()),
                "其中实际有调价样本数": int(threshold["实际是否有调价"].sum()),
                "实际未调比例": float(threshold["实际是否未调"].mean()) if len(threshold) else None,
                "说明": "product_oil.csv只包含调价公告日期，无法直接观测所有未发布调价公告的10工作日窗口；本表仅验证公告窗口中的<50元样本。",
            }
        ]
    )
    summary.to_csv(THRESHOLD_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return summary.iloc[0].to_dict()


def make_two_bar_svg(
    summary: pd.DataFrame,
    label_column: str,
    value_column: str,
    path: Path,
    title: str,
    note: str,
) -> None:
    width = 760
    height = 420
    left = 90
    right = 40
    top = 70
    bottom = 80
    chart_height = height - top - bottom
    chart_width = width - left - right
    values = summary[value_column].fillna(0).to_numpy(dtype=float)
    vmax = max(float(np.nanmax(values)), 1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        "</style>",
        f'<text class="title" x="{left}" y="28">{html.escape(title)}</text>',
        f'<text class="note" x="{left}" y="48">{html.escape(note)}</text>',
    ]
    for tick in np.linspace(0, vmax, 6):
        y = top + chart_height - tick / vmax * chart_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{tick:.2f}</text>')

    bar_width = 120
    gap = 90
    start_x = left + 130
    colors = ["#2f6fbb", "#c35a4a"]
    for idx, row in summary.reset_index(drop=True).iterrows():
        value = float(row[value_column]) if pd.notna(row[value_column]) else 0.0
        bar_height = value / vmax * chart_height
        x = start_x + idx * (bar_width + gap)
        y = top + chart_height - bar_height
        label = html.escape(str(row[label_column]))
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[idx % len(colors)]}" rx="3"/>')
        parts.append(f'<text x="{x + bar_width/2}" y="{height-50}" text-anchor="middle">{label}</text>')
        parts.append(f'<text x="{x + bar_width/2}" y="{y-8:.1f}" text-anchor="middle">{value:.3f}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top+chart_height}" x2="{width-right}" y2="{top+chart_height}"/>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def make_interval_svg(summary: pd.DataFrame, path: Path) -> None:
    width = 860
    height = 460
    left = 96
    right = 50
    top = 70
    bottom = 92
    chart_height = height - top - bottom
    chart_width = width - left - right
    values = summary["平均传递系数"].fillna(0).to_numpy(dtype=float)
    vmax = max(float(np.nanmax(values)), 1.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;font-size:12px;fill:#222}",
        ".title{font-size:18px;font-weight:700}",
        ".note{fill:#666;font-size:11px}",
        ".axis{stroke:#222;stroke-width:1}",
        ".grid{stroke:#ddd;stroke-width:1}",
        "</style>",
        f'<text class="title" x="{left}" y="28">不同国际油价区间的平均传递系数</text>',
        f'<text class="note" x="{left}" y="48">区间按本期10工作日一揽子均值（USD/bbl）划分；空缺或小样本区间需谨慎解释。</text>',
    ]
    for tick in np.linspace(0, vmax, 6):
        y = top + chart_height - tick / vmax * chart_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{tick:.2f}</text>')

    bar_width = 90
    gap = (chart_width - 4 * bar_width) / 3
    colors = ["#78909c", "#2f6fbb", "#8a6fb5", "#c35a4a"]
    for idx, row in summary.reset_index(drop=True).iterrows():
        value = row["平均传递系数"]
        value = float(value) if pd.notna(value) else 0.0
        bar_height = value / vmax * chart_height
        x = left + idx * (bar_width + gap)
        y = top + chart_height - bar_height
        label = html.escape(str(row["油价区间"]))
        sample_count = int(row["样本数"])
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[idx]}" rx="3"/>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{height-58}" text-anchor="middle">{label}</text>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{height-40}" text-anchor="middle">n={sample_count}</text>')
        if sample_count:
            parts.append(f'<text x="{x + bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle">{value:.3f}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top+chart_height}" x2="{width-right}" y2="{top+chart_height}"/>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(asymmetry: dict[str, object], interval: dict[str, object], threshold: dict[str, object]) -> None:
    report = {
        "analysis_scope": "2016-01-01以后，且能匹配理论成本10工作日均值的汽油调价窗口",
        "theoretical_adjustment_definition": "理论应调幅度 = 理论国内成本_CNY_per_ton_本期10工作日均值 - 上期10工作日均值",
        "actual_adjustment_definition": "实际调价幅度 = product_oil.csv中的汽油涨跌真实值_CNY_per_ton",
        "asymmetry": asymmetry,
        "interval_performance": interval,
        "threshold_50_cny_per_ton": threshold,
        "outputs": {
            "asymmetry_samples": str(ASYMMETRY_SAMPLES_PATH),
            "asymmetry_summary": str(ASYMMETRY_SUMMARY_PATH),
            "asymmetry_svg": str(ASYMMETRY_SVG_PATH),
            "interval_summary": str(INTERVAL_SUMMARY_PATH),
            "interval_samples": str(INTERVAL_SAMPLES_PATH),
            "interval_svg": str(INTERVAL_SVG_PATH),
            "threshold_table": str(THRESHOLD_TABLE_PATH),
            "threshold_summary": str(THRESHOLD_SUMMARY_PATH),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    df = read_base_data()
    asymmetry = run_asymmetry_analysis(df)
    interval = run_interval_analysis(df)
    threshold = run_threshold_analysis(df)
    write_report(asymmetry, interval, threshold)

    print(f"Base rows after 2016 with theoretical windows: {len(df)}")
    effective = asymmetry["effective_abs_theoretical_change_ge_50"]
    print(f"Effective asymmetry index: {effective['asymmetry_index']:.6f}")
    print(f"Conclusion: {asymmetry['preferred_conclusion']}")
    print(f"Threshold <50 samples: {threshold['理论调幅绝对值小于50样本数']}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
