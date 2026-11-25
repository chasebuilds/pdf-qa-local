// Browser benchmark: runs the real page (pdf.js + MiniLM + DistilBERT QA) in headless Chromium
// on the synthetic corpus from bench/synth.py. First: python -m bench.export_questions --seed N
// node web/bench.mjs --seed 2 [--limit 30] [--out bench/results-web-seed2.json]
import { chromium } from "playwright";
import { readFile, writeFile } from "node:fs/promises";
import { serve } from "./serve.mjs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const seed = Number(arg("seed", 2));
const limit = Number(arg("limit", 0));
const out = arg("out", null);
const port = Number(process.env.PORT ?? 4392);

const dataUrl = new URL(`../bench/data/seed${seed}/`, import.meta.url);
let qas = JSON.parse(await readFile(new URL("questions.json", dataUrl), "utf-8"));
const answerable = qas.filter((q) => q.answer !== null);
const refusals = qas.filter((q) => q.answer === null);
let picked = answerable;
if (limit) {
  const step = Math.max(Math.floor(answerable.length / limit), 1);
  picked = answerable.filter((_, i) => i % step === 0).slice(0, limit);
}

const server = await serve(port);
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.goto(`http://127.0.0.1:${port}/web/?bench`);
  await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 600_000 });
  if (await page.evaluate(() => document.body.dataset.ready) !== "1") throw new Error("page failed to start");
  const files = [...new Set(answerable.map((q) => q.file))].sort();
  const t0 = Date.now();
  await page.evaluate((urls) => window.pdfqa.loadUrls(urls), files.map((f) => `/bench/data/seed${seed}/${f}`));
  const indexS = (Date.now() - t0) / 1000;
  const ask = (q) => page.evaluate((q) => window.pdfqa.ask(q).then((r) => ({ ...r, hits: r.hits.map((h) => [h.chunk.file, h.chunk.page]) })), q);

  let hit1 = 0, hitk = 0, ansOk = 0, citeOk = 0, falseRefusal = 0, refusedOk = 0;
  const misses = [];
  const t1 = Date.now();
  for (const q of picked) {
    const r = await ask(q.question);
    const gold = `${q.file}|${q.page}`;
    const locs = r.hits.map(([f, p]) => `${f}|${p}`);
    hit1 += locs[0] === gold;
    hitk += locs.includes(gold);
    falseRefusal += !r.found;
    const text = r.found ? `${r.answer} ${r.sentence}` : "";
    const answerHas = r.found && r.answer.toLowerCase().includes(q.answer.toLowerCase());
    ansOk += answerHas;
    const cited = r.found && `${r.file}|${r.page}` === gold;
    citeOk += cited;
    if (!cited || !answerHas) misses.push({ q: q.question, gold, got: r.found ? `${r.file}|${r.page}: ${r.answer}` : "refused", text });
  }
  const wrongAnswers = [];
  for (const q of refusals) {
    const r = await ask(q.question);
    refusedOk += !r.found;
    if (r.found) wrongAnswers.push({ q: q.question, got: `${r.file}|${r.page}: ${r.answer}` });
  }
  const askS = (Date.now() - t1) / 1000;
  const n = picked.length;
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const result = {
    embedder: "transformers.js:Xenova/all-MiniLM-L6-v2 (q8)",
    llm: "transformers.js:Xenova/distilbert-base-cased-distilled-squad (q8), extractive",
    seed,
    documents: files.length,
    answerable_questions: n,
    unanswerable_questions: refusals.length,
    retrieval_hit_at_5: r3(hitk / n),
    retrieval_hit_at_1: r3(hit1 / n),
    answer_contains_fact: r3(ansOk / n),
    citation_correct: r3(citeOk / n),
    false_refusal_rate: r3(falseRefusal / n),
    correct_refusal_rate: r3(refusedOk / refusals.length),
    index_seconds: Math.round(indexS * 10) / 10,
    ask_seconds_per_question: Math.round((askS / (n + refusals.length)) * 100) / 100,
  };
  console.log(JSON.stringify(result, null, 2));
  if (process.argv.includes("--verbose")) console.log(JSON.stringify({ misses, wrongAnswers }, null, 1));
  if (out) await writeFile(out, JSON.stringify([result], null, 2) + "\n");
} finally {
  await browser.close();
  server.close();
}
