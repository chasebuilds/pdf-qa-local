import { test, expect } from "@playwright/test";

test("sample question gets a cited answer, unknown product is refused", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(page.locator("body")).toHaveAttribute("data-ready", "1", { timeout: 240_000 });
  await expect(page.locator("#docs")).toContainText("manual_02.pdf");

  await page.fill("#q", "What is the maximum pressure of the Zinorin 400 unit?");
  await page.click("#ask");
  await expect(page.locator("#answer-text")).toContainText("12 bar");
  await expect(page.locator("#answer-cite")).toHaveText("manual_02.pdf p.3");
  if (process.env.SCREENSHOT) await page.screenshot({ path: process.env.SCREENSHOT, fullPage: true });

  await page.fill("#q", "What is the maximum pressure of the Quorvex 900 unit?");
  await page.click("#ask");
  await expect(page.locator("#answer-text")).toHaveText("Not in these documents.");
  expect(errors).toEqual([]);
});
