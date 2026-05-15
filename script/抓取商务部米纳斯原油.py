from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datas"

NATIONAL_OIL_PATH = DATA_DIR / "national_oil.csv"
MOFCOM_MINAS_PATH = DATA_DIR / "商务部米纳斯原油.csv"
MERGED_MINAS_PATH = DATA_DIR / "米纳斯原油_商务部与历史拼接.csv"
FULL_MERGED_OIL_PATH = DATA_DIR / "国际原油价格_米纳斯商务部补全.csv"
SUMMARY_PATH = DATA_DIR / "米纳斯原油拼接摘要.json"

SEQNO = "184"
SOURCE_SWITCH_DATE = pd.Timestamp("2020-01-01")
ENDPOINT = "https://price.mofcom.gov.cn/datamofcom/front/price/pricequotation/priceQueryList"
REFERER = (
    "https://price.mofcom.gov.cn/price_2021/pricequotation/"
    f"pricequotationdetail.shtml?seqno={SEQNO}"
)
PAGE_SIZE = 2000


def fetch_page(page_number: int) -> dict:
    try:
        response = requests.post(
            ENDPOINT,
            data={
                "seqno": SEQNO,
                "startTime": "",
                "endTime": "",
                "pageNumber": str(page_number),
                "pageSize": str(PAGE_SIZE),
            },
            headers={
                "Referer": REFERER,
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.SSLError:
        return fetch_page_with_powershell(page_number)


def fetch_page_with_powershell(page_number: int) -> dict:
    command = f"""
$body = @{{
  seqno = '{SEQNO}'
  startTime = ''
  endTime = ''
  pageNumber = '{page_number}'
  pageSize = '{PAGE_SIZE}'
}}
$headers = @{{
  Referer = '{REFERER}'
  'X-Requested-With' = 'XMLHttpRequest'
  'User-Agent' = 'Mozilla/5.0'
}}
$result = Invoke-RestMethod -Uri '{ENDPOINT}' -Method Post -Body $body -Headers $headers -TimeoutSec 30
$result | ConvertTo-Json -Depth 12 -Compress
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def fetch_mofcom_minas() -> pd.DataFrame:
    first_page = fetch_page(1)
    rows = list(first_page.get("rows", []))
    max_page = int(first_page.get("maxPageNum", 1))

    for page_number in range(2, max_page + 1):
        page = fetch_page(page_number)
        rows.extend(page.get("rows", []))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("商务部接口没有返回米纳斯原油数据")

    df["日期"] = pd.to_datetime(
        df["yyyy"].astype(str) + "-" + df["mm"].astype(str) + "-" + df["dd"].astype(str),
        errors="coerce",
    )
    df["米纳斯"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["日期", "米纳斯"])
    df["来源"] = "商务部"
    df = df.rename(
        columns={
            "prod_name": "商品名称",
            "prod_spec": "规格",
            "unit": "单位",
            "region": "地区",
        }
    )
    columns = ["日期", "米纳斯", "来源", "商品名称", "规格", "单位", "地区"]
    return df[columns].drop_duplicates(subset=["日期"], keep="first").sort_values(
        "日期",
        ascending=False,
    )


def read_historical_minas() -> pd.DataFrame:
    historical = pd.read_csv(NATIONAL_OIL_PATH)
    historical["日期"] = pd.to_datetime(historical["日期"], errors="coerce")
    historical["米纳斯"] = pd.to_numeric(historical["米纳斯"], errors="coerce")
    historical = historical.dropna(subset=["日期", "米纳斯"])
    historical = historical[historical["日期"] < SOURCE_SWITCH_DATE].copy()
    historical["来源"] = "national_oil.csv"
    return historical[["日期", "米纳斯", "来源"]]


def build_full_merged_oil(mofcom: pd.DataFrame) -> pd.DataFrame:
    national = pd.read_csv(NATIONAL_OIL_PATH)
    national["日期"] = pd.to_datetime(national["日期"], errors="coerce")
    national = national.dropna(subset=["日期"])

    mofcom_prices = mofcom[["日期", "米纳斯"]].rename(
        columns={"米纳斯": "商务部米纳斯"}
    )
    full = pd.merge(national, mofcom_prices, on="日期", how="outer")
    full["米纳斯"] = pd.to_numeric(full["米纳斯"], errors="coerce")
    full["商务部米纳斯"] = pd.to_numeric(full["商务部米纳斯"], errors="coerce")
    use_mofcom = (full["日期"] >= SOURCE_SWITCH_DATE) & full["商务部米纳斯"].notna()
    use_national = (full["日期"] < SOURCE_SWITCH_DATE) & full["米纳斯"].notna()
    full.loc[use_mofcom, "米纳斯"] = full.loc[use_mofcom, "商务部米纳斯"]
    full["米纳斯来源"] = ""
    full.loc[use_mofcom, "米纳斯来源"] = "商务部"
    full.loc[use_national, "米纳斯来源"] = "national_oil.csv"
    full = full.drop(columns=["商务部米纳斯"])

    original_columns = list(pd.read_csv(NATIONAL_OIL_PATH, nrows=0).columns)
    columns = original_columns + ["米纳斯来源"]
    return full[columns].sort_values("日期", ascending=False).reset_index(drop=True)


def format_date(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def write_outputs(
    mofcom: pd.DataFrame,
    merged: pd.DataFrame,
    full_merged: pd.DataFrame,
) -> None:
    mofcom_output = mofcom.copy()
    mofcom_output["日期"] = mofcom_output["日期"].map(format_date)
    mofcom_output.to_csv(MOFCOM_MINAS_PATH, index=False, encoding="utf-8-sig")

    merged_output = merged.copy()
    merged_output["日期"] = merged_output["日期"].map(format_date)
    merged_output.to_csv(MERGED_MINAS_PATH, index=False, encoding="utf-8-sig")

    full_output = full_merged.copy()
    full_output["日期"] = full_output["日期"].map(format_date)
    full_output.to_csv(FULL_MERGED_OIL_PATH, index=False, encoding="utf-8-sig")

    source_counts = merged["来源"].value_counts().to_dict()
    summary = {
        "seqno": SEQNO,
        "source_url": REFERER,
        "rule": "2020-01-01及之后使用商务部米纳斯原油；2020-01-01之前使用national_oil.csv米纳斯列",
        "source_switch_date": SOURCE_SWITCH_DATE.strftime("%Y-%m-%d"),
        "mofcom_rows": int(len(mofcom)),
        "merged_rows": int(len(merged)),
        "full_merged_rows": int(len(full_merged)),
        "merged_date_min": merged["日期"].min().strftime("%Y-%m-%d"),
        "merged_date_max": merged["日期"].max().strftime("%Y-%m-%d"),
        "merged_rows_by_source": {key: int(value) for key, value in source_counts.items()},
        "outputs": {
            "商务部米纳斯原油": str(MOFCOM_MINAS_PATH),
            "米纳斯原油_商务部与历史拼接": str(MERGED_MINAS_PATH),
            "国际原油价格_米纳斯商务部补全": str(FULL_MERGED_OIL_PATH),
            "摘要": str(SUMMARY_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    mofcom = fetch_mofcom_minas()
    historical = read_historical_minas()
    mofcom_for_merge = mofcom[mofcom["日期"] >= SOURCE_SWITCH_DATE][
        ["日期", "米纳斯", "来源"]
    ].copy()
    merged = pd.concat([mofcom_for_merge, historical], ignore_index=True)
    merged = merged.drop_duplicates(subset=["日期"], keep="first")
    merged = merged.sort_values("日期", ascending=False).reset_index(drop=True)
    full_merged = build_full_merged_oil(mofcom)
    write_outputs(mofcom, merged, full_merged)

    print(f"商务部米纳斯行数: {len(mofcom)}")
    print(f"拼接后行数: {len(merged)}")
    print(
        "拼接后日期范围: "
        f"{merged['日期'].min().date()} to {merged['日期'].max().date()}"
    )
    print(f"Saved: {MOFCOM_MINAS_PATH}")
    print(f"Saved: {MERGED_MINAS_PATH}")
    print(f"Saved: {FULL_MERGED_OIL_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
