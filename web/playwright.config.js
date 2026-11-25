import { defineConfig } from "@playwright/test";

// BASE_URL points the smoke test at the live Pages site; otherwise a local server on 4393.
const live = process.env.BASE_URL;
export default defineConfig({
  testDir: "tests",
  timeout: 300_000,
  use: { baseURL: live ?? "http://127.0.0.1:4393/web/" },
  webServer: live ? undefined : { command: "node serve.mjs", env: { PORT: "4393" }, url: "http://127.0.0.1:4393/web/", reuseExistingServer: false },
});
