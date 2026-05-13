from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datas"
OUTPUT_DIR = DATA_DIR / "output"

OLD_USD_CNY = DATA_DIR / "USD_CNY.csv"
NEW_USD_CNY = DATA_DIR / "USD_CNY历史数据.csv"
MERGED_USD_CNY = DATA_DIR / "USD_CNY.csv"
SUMMARY_PATH = OUTPUT_DIR / "USD_CNY_merge_summary.json"

DATE_COLUMN = "日期"
NUMERIC_COLUMNS = ["收盘", "开盘", "高", "低"]


def read_fx(path: Path, source: str, priority: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"{path} 缺少日期列")

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN].astype(str).str.strip(),
        errors="coerce",
    )
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["_source"] = source
    df["_priority"] = priority
    df = df.dropna(subset=[DATE_COLUMN])
    return df


def format_date_no_padding(value: pd.Timestamp) -> str:
    return f"{value.year}-{value.month}-{value.day}"


def merge_usd_cny() -> tuple[pd.DataFrame, dict[str, int], int, int]:
    old_df = read_fx(OLD_USD_CNY, "USD_CNY.csv", priority=1)
    new_df = read_fx(NEW_USD_CNY, "USD_CNY历史数据.csv", priority=2)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.sort_values([DATE_COLUMN, "_priority"])
    merged = merged.drop_duplicates(subset=[DATE_COLUMN], keep="last")
    merged = merged.sort_values(DATE_COLUMN, ascending=False).reset_index(drop=True)

    source_counts = merged["_source"].value_counts().to_dict()
    merged = merged.drop(columns=["_source", "_priority"])

    # Keep the original display style used by the downloaded CSV files.
    merged[DATE_COLUMN] = merged[DATE_COLUMN].map(format_date_no_padding)
    return merged, source_counts, len(old_df), len(new_df)


def write_summary(
    merged: pd.DataFrame,
    source_counts: dict[str, int],
    old_rows: int,
    new_rows: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(merged[DATE_COLUMN], errors="coerce")
    summary = {
        "old_file": str(OLD_USD_CNY),
        "new_file": str(NEW_USD_CNY),
        "merged_file": str(MERGED_USD_CNY),
        "old_rows": old_rows,
        "new_rows": new_rows,
        "merged_rows": int(len(merged)),
        "date_min": dates.min().strftime("%Y-%m-%d"),
        "date_max": dates.max().strftime("%Y-%m-%d"),
        "duplicate_dates_after_merge": int(dates.duplicated().sum()),
        "overlap_rule": "same 日期 keeps USD_CNY历史数据.csv over USD_CNY.csv",
        "kept_rows_by_source": source_counts,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    merged, source_counts, old_rows, new_rows = merge_usd_cny()
    merged.to_csv(MERGED_USD_CNY, index=False, encoding="utf-8-sig")
    write_summary(merged, source_counts, old_rows, new_rows)

    dates = pd.to_datetime(merged[DATE_COLUMN], errors="coerce")
    print(f"Old rows: {old_rows}")
    print(f"New rows: {new_rows}")
    print(f"Merged rows: {len(merged)}")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"Saved: {MERGED_USD_CNY}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
