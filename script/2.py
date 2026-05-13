from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datas"
OUTPUT_DIR = DATA_DIR / "output"

PRODUCT_OIL_PATH = DATA_DIR / "product_oil.csv"
BASKET_COST_PATH = OUTPUT_DIR / "basket_cost_model_ready.csv"
OUTPUT_PATH = OUTPUT_DIR / "gasoline_cost_10workday_change.csv"
DETAIL_OUTPUT_PATH = OUTPUT_DIR / "gasoline_cost_10workday_change_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "gasoline_cost_10workday_change_summary.json"

ROLLING_WORKDAYS = 10
ALIGN_TOLERANCE_DAYS = 10


def parse_price(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    multiplier = 10000 if "万" in text else 1
    number_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not number_match:
        return None

    return float(number_match.group()) * multiplier


def parse_change(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    sign = -1 if "↓" in text or "-" in text else 1
    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if not number_match:
        return None

    return sign * float(number_match.group())


def read_gasoline() -> pd.DataFrame:
    product = pd.read_csv(PRODUCT_OIL_PATH)
    gasoline = product[["调整日期", "汽油", "汽油涨跌"]].copy()
    gasoline["调整日期"] = pd.to_datetime(gasoline["调整日期"], errors="coerce")
    gasoline["汽油价格真实值_CNY_per_ton"] = gasoline["汽油"].map(parse_price)
    gasoline["汽油涨跌真实值_CNY_per_ton"] = gasoline["汽油涨跌"].map(parse_change)
    gasoline = gasoline.dropna(subset=["调整日期", "汽油价格真实值_CNY_per_ton"])
    return gasoline.sort_values("调整日期").reset_index(drop=True)


def read_cost_basis() -> pd.DataFrame:
    cost = pd.read_csv(BASKET_COST_PATH)
    cost["日期"] = pd.to_datetime(cost["日期"], errors="coerce")
    cost["国内成本基础_CNY_per_ton"] = pd.to_numeric(
        cost["国内成本基础_CNY_per_ton"],
        errors="coerce",
    )
    cost = cost.dropna(subset=["日期", "国内成本基础_CNY_per_ton"])
    cost = cost.sort_values("日期").reset_index(drop=True)

    cost["理论国内成本基础_本期10工作日均值_CNY_per_ton"] = (
        cost["国内成本基础_CNY_per_ton"]
        .rolling(window=ROLLING_WORKDAYS, min_periods=ROLLING_WORKDAYS)
        .mean()
    )
    cost["10工作日窗口开始日期"] = cost["日期"].shift(ROLLING_WORKDAYS - 1)
    cost["10工作日窗口结束日期"] = cost["日期"]
    return cost


def align_cost_to_adjustment_dates(
    gasoline: pd.DataFrame,
    cost: pd.DataFrame,
) -> pd.DataFrame:
    cost_for_merge = cost[
        [
            "日期",
            "国内成本基础_CNY_per_ton",
            "理论国内成本基础_本期10工作日均值_CNY_per_ton",
            "10工作日窗口开始日期",
            "10工作日窗口结束日期",
        ]
    ].dropna(subset=["理论国内成本基础_本期10工作日均值_CNY_per_ton"])

    result = pd.merge_asof(
        gasoline.sort_values("调整日期"),
        cost_for_merge.sort_values("日期"),
        left_on="调整日期",
        right_on="日期",
        direction="backward",
        tolerance=pd.Timedelta(days=ALIGN_TOLERANCE_DAYS),
    )
    return result.rename(columns={"日期": "匹配成本日期"})


def add_change_rates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("调整日期").copy()

    result["理论国内成本基础_上期10工作日均值_CNY_per_ton"] = result[
        "理论国内成本基础_本期10工作日均值_CNY_per_ton"
    ].shift(1)
    result["理论国内成本基础涨跌值_CNY_per_ton"] = (
        result["理论国内成本基础_本期10工作日均值_CNY_per_ton"]
        - result["理论国内成本基础_上期10工作日均值_CNY_per_ton"]
    )
    result["理论国内成本变化率"] = (
        result["理论国内成本基础_本期10工作日均值_CNY_per_ton"]
        / result["理论国内成本基础_上期10工作日均值_CNY_per_ton"]
        - 1
    )

    result["上期汽油价格真实值_CNY_per_ton"] = result["汽油价格真实值_CNY_per_ton"].shift(1)
    result["汽油价格变化率"] = (
        result["汽油价格真实值_CNY_per_ton"]
        / result["上期汽油价格真实值_CNY_per_ton"]
        - 1
    )

    denominator = result["理论国内成本变化率"].replace(0, pd.NA)
    result["汽油变化率_理论国内成本变化率比值"] = result["汽油价格变化率"] / denominator
    return result


def build_display_table(result: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "调整日期",
        "汽油价格真实值_CNY_per_ton",
        "汽油涨跌真实值_CNY_per_ton",
        "理论国内成本基础_本期10工作日均值_CNY_per_ton",
        "理论国内成本基础_上期10工作日均值_CNY_per_ton",
        "理论国内成本基础涨跌值_CNY_per_ton",
        "汽油价格变化率",
        "理论国内成本变化率",
        "汽油变化率_理论国内成本变化率比值",
        "匹配成本日期",
        "10工作日窗口开始日期",
        "10工作日窗口结束日期",
    ]
    return result[columns].copy()


def format_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in date_columns:
        formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
    return formatted


def write_outputs(result: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail = result.sort_values("调整日期", ascending=False).copy()
    display = build_display_table(detail)

    date_columns = [
        "调整日期",
        "匹配成本日期",
        "10工作日窗口开始日期",
        "10工作日窗口结束日期",
    ]
    detail = format_dates(detail, date_columns)
    display = format_dates(display, date_columns)

    display.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    detail.to_csv(DETAIL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    matched = result["理论国内成本基础_本期10工作日均值_CNY_per_ton"].notna()
    summary = {
        "product_file": str(PRODUCT_OIL_PATH),
        "basket_cost_file": str(BASKET_COST_PATH),
        "output_file": str(OUTPUT_PATH),
        "detail_output_file": str(DETAIL_OUTPUT_PATH),
        "rolling_workdays": ROLLING_WORKDAYS,
        "align_rule": (
            "for each gasoline adjustment date, use the nearest previous "
            "basket_cost_model_ready date within 10 calendar days"
        ),
        "input_gasoline_rows": int(len(result)),
        "matched_cost_rows": int(matched.sum()),
        "unmatched_cost_rows": int((~matched).sum()),
        "date_min": result["调整日期"].min().strftime("%Y-%m-%d"),
        "date_max": result["调整日期"].max().strftime("%Y-%m-%d"),
        "matched_date_min": (
            result.loc[matched, "调整日期"].min().strftime("%Y-%m-%d")
            if matched.any()
            else None
        ),
        "matched_date_max": (
            result.loc[matched, "调整日期"].max().strftime("%Y-%m-%d")
            if matched.any()
            else None
        ),
        "change_rate_definition": {
            "理论国内成本基础涨跌值_CNY_per_ton": "current 10-workday average - previous adjustment's 10-workday average",
            "理论国内成本变化率": "current 10-workday average / previous adjustment's 10-workday average - 1",
            "汽油价格变化率": "current gasoline CNY/ton / previous adjustment gasoline CNY/ton - 1",
            "汽油变化率_理论国内成本变化率比值": "汽油价格变化率 / 理论国内成本变化率",
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    gasoline = read_gasoline()
    cost = read_cost_basis()
    aligned = align_cost_to_adjustment_dates(gasoline, cost)
    result = add_change_rates(aligned)
    write_outputs(result)

    matched = result["理论国内成本基础_本期10工作日均值_CNY_per_ton"].notna()
    print(f"Gasoline rows: {len(result)}")
    print(f"Matched cost rows: {matched.sum()}")
    print(f"Unmatched cost rows: {(~matched).sum()}")
    print(f"Saved display table: {OUTPUT_PATH}")
    print(f"Saved detail table: {DETAIL_OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
