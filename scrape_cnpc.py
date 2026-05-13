"""
CNPC crude-oil scraper using page.evaluate approach.
数据直接发送到远程端点。
"""

from playwright.sync_api import sync_playwright

NATIONAL = "https://www.cnpc.com.cn/cnpcstockinfo/oilgas/scgc_crudeoil_more.jsp"

SCRIPT = """
const totalText = await page.locator('body').innerText();
const totalPagesMatch = totalText.match(/共(\d+)页/);
if (!totalPagesMatch) throw new Error('cannot find total pages');
const totalPages = Number(totalPagesMatch[1]);
const endpoint = 'http://127.0.0.1:8765/ingest';
const finishEndpoint = 'http://127.0.0.1:8765/done';

const extractRows = () => Array.from(document.querySelectorAll('table.datagrid-btable tr'))
  .map((tr) => {
    const cells = Array.from(tr.querySelectorAll('td[field]'));
    if (!cells.length) return null;
    const row = {};
    for (const cell of cells) row[cell.getAttribute('field')] = cell.textContent.trim();
    return row;
  })
  .filter(Boolean);

const normalize = (row) => ({
  日期: row.updatetime || '',
  WTI: row.wti || '',
  布伦特: row.blt || '',
  迪拜: row.dib || '',
  阿曼: row.omn || '',
  塔皮斯: row.tps || '',
  米纳斯: row.mns || '',
  杜里: row.dul || '',
  辛塔: row.cnt || '',
  大庆: row.dqn || '',
  胜利: row.sli || '',
  欧佩克: row.opec || '',
  ESPO: row.espo || '',
});

async function sendRows(pageNo, rows) {
  return await page.evaluate(async ({ endpoint, pageNo, rows }) => {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pageNo, rows }),
    });
    if (!resp.ok) throw new Error(`upload failed ${resp.status}`);
    return await resp.json();
  }, { endpoint, pageNo, rows });
}

const result = { totalPages, totalRows: 0 };
for (let pageNo = 1; pageNo <= totalPages; pageNo++) {
  if (pageNo > 1) {
    await page.evaluate((n) => jumpPage(String(n)), pageNo);
    await page.waitForFunction((n) => document.querySelector('#toPageNo')?.value === String(n), pageNo, { timeout: 20000 });
    await page.waitForFunction(() => document.querySelectorAll('table.datagrid-btable tr').length > 1, { timeout: 20000 });
  }
  const rows = await page.evaluate(extractRows);
  if (!rows.length) throw new Error(`no rows on page ${pageNo}`);
  const normalized = rows.map(normalize);
  await sendRows(pageNo, normalized);
  result.totalRows += normalized.length;
  if (pageNo % 50 === 0 || pageNo === totalPages) {
    console.log(`sent ${pageNo}/${totalPages}`);
  }
}
await page.evaluate(async ({ finishEndpoint }) => {
  await fetch(finishEndpoint, { method: 'POST' });
}, { finishEndpoint });
return result;
"""


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        try:
            page = browser.new_page()
            page.goto(NATIONAL, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)

            # Check for verification
            body_text = page.locator("body").inner_text()
            if any(k in body_text for k in ("验证码", "安全验证", "verify", "verification")):
                print("出现验证码，请在浏览器里完成后按回车继续...")
                input()

            # Wait for table
            page.wait_for_function(
                "() => document.querySelectorAll('table.datagrid-btable tr').length > 1",
                timeout=15000,
            )

            # Execute the script
            result = page.evaluate(SCRIPT)
            print(f"完成！共 {result['totalPages']} 页，{result['totalRows']} 行数据已发送")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
