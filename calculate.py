from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datas"
OUTPUT_DIR = DATA_DIR / "output"

NATIONAL_OIL_PATH = DATA_DIR / "national_oil.csv"
USD_CNY_PATH = DATA_DIR / "USD_CNY.csv"

# 题目这一阶段要求复现“一揽子价格”。这里按题设给定权重：
# Brent: Dubai: Minas = 0.4: 0.3: 0.3。
BASKET_WEIGHTS = {
    "布伦特": 0.4,
    "迪拜": 0.3,
    "米纳斯": 0.3,
}

# 国际原油以美元/桶计价；国内成本基础需要人民币/吨。
# 不同油种密度不同，桶吨换算会略有差异。这里先采用原油常用近似：
# 1 metric ton ~= 7.33 barrels。
BARRELS_PER_TON = 7.33

# 汇率只允许向前沿用短期交易日缺口，避免把 USD_CNY.csv 末尾的旧汇率
# 长期外推到没有汇率覆盖的年份。
FX_MAX_STALE_DAYS = 7


def read_national_oil() -> pd.DataFrame:
    df = pd.read_csv(NATIONAL_OIL_PATH)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")

    for column in BASKET_WEIGHTS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    return df


def read_usd_cny() -> pd.DataFrame:
    df = pd.read_csv(USD_CNY_PATH)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["USD_CNY"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df.dropna(subset=["日期", "USD_CNY"])
    df["汇率日期"] = df["日期"]
    df = df[["汇率日期", "USD_CNY"]].sort_values("汇率日期").reset_index(drop=True)
    return df


def add_basket_price(oil: pd.DataFrame) -> pd.DataFrame:
    result = oil.copy()
    result["一揽子价格_USD_per_bbl"] = sum(
        result[column] * weight for column, weight in BASKET_WEIGHTS.items()
    )
    result["一揽子价格_油价完整"] = result[list(BASKET_WEIGHTS)].notna().all(axis=1)
    return result


def merge_exchange_rate(oil: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    # 外汇数据通常只在交易日有值。对油价日期使用“向前取最近已知汇率”
    # 可以避免周末、节假日导致的大量空值，也符合历史模拟不能使用未来信息的原则。
    merged = pd.merge_asof(
        oil.sort_values("日期"),
        fx.sort_values("汇率日期"),
        left_on="日期",
        right_on="汇率日期",
        direction="backward",
        tolerance=pd.Timedelta(days=FX_MAX_STALE_DAYS),
    )
    merged["汇率可得"] = merged["USD_CNY"].notna()
    merged["汇率间隔天数"] = (merged["日期"] - merged["汇率日期"]).dt.days
    return merged


def add_domestic_cost_basis(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["国内成本基础_CNY_per_ton"] = (
        result["一揽子价格_USD_per_bbl"] * result["USD_CNY"] * BARRELS_PER_TON
    )
    result["可用于建模"] = (
        result["一揽子价格_油价完整"]
        & result["汇率可得"]
        & result["国内成本基础_CNY_per_ton"].notna()
    )
    return result


def write_outputs(result: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_columns = [
        "日期",
        "布伦特",
        "迪拜",
        "米纳斯",
        "一揽子价格_USD_per_bbl",
        "汇率日期",
        "USD_CNY",
        "汇率间隔天数",
        "国内成本基础_CNY_per_ton",
        "一揽子价格_油价完整",
        "汇率可得",
        "可用于建模",
    ]
    model_columns = [
        "日期",
        "布伦特",
        "迪拜",
        "米纳斯",
        "一揽子价格_USD_per_bbl",
        "汇率日期",
        "USD_CNY",
        "汇率间隔天数",
        "国内成本基础_CNY_per_ton",
    ]

    full = result[full_columns].sort_values("日期", ascending=False).copy()
    usable = (
        result.loc[result["可用于建模"], model_columns]
        .sort_values("日期", ascending=False)
        .copy()
    )

    for df in [full, usable]:
        df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")
        df["汇率日期"] = df["汇率日期"].dt.strftime("%Y-%m-%d")

    full.to_csv(OUTPUT_DIR / "basket_cost_full.csv", index=False, encoding="utf-8-sig")
    usable.to_csv(
        OUTPUT_DIR / "basket_cost_model_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "basket_weights": BASKET_WEIGHTS,
        "barrels_per_ton": BARRELS_PER_TON,
        "formula": (
            "domestic_cost_cny_per_ton = "
            "(0.4 * Brent + 0.3 * Dubai + 0.3 * Minas) * USD_CNY * 7.33"
        ),
        "exchange_rate_alignment": (
            "merge_asof backward: use nearest previous USD_CNY close within "
            f"{FX_MAX_STALE_DAYS} days"
        ),
        "exchange_rate_max_stale_days": FX_MAX_STALE_DAYS,
        "input_rows": int(len(result)),
        "basket_valid_rows": int(result["一揽子价格_油价完整"].sum()),
        "fx_available_rows": int(result["汇率可得"].sum()),
        "model_ready_rows": int(result["可用于建模"].sum()),
        "full_date_min": full["日期"].min() if not full.empty else None,
        "full_date_max": full["日期"].max() if not full.empty else None,
        "model_ready_date_min": usable["日期"].min() if not usable.empty else None,
        "model_ready_date_max": usable["日期"].max() if not usable.empty else None,
        "outputs": {
            "full": str(OUTPUT_DIR / "basket_cost_full.csv"),
            "model_ready": str(OUTPUT_DIR / "basket_cost_model_ready.csv"),
        },
    }
    (OUTPUT_DIR / "basket_cost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    oil = read_national_oil()
    fx = read_usd_cny()
    basket = add_basket_price(oil)
    merged = merge_exchange_rate(basket, fx)
    result = add_domestic_cost_basis(merged)
    write_outputs(result)

    usable = result[result["可用于建模"]]
    print(f"Full rows: {len(result)}")
    print(f"Model-ready rows: {len(usable)}")
    if not usable.empty:
        print(
            "Model-ready date range: "
            f"{usable['日期'].min().date()} to {usable['日期'].max().date()}"
        )
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
