import * as pdfjs from "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.min.mjs";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
import { NOT_FOUND, Retriever, answer, chunkPages } from "./pipeline.js";

pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.worker.min.mjs";
env.allowLocalModels = false;

const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
const QA_MODEL = "Xenova/distilbert-base-cased-distilled-squad";
const SAMPLE_BASE = location.pathname.includes("/web/") ? "../samples/" : "samples/";
const SAMPLES = ["manual_01.pdf", "manual_02.pdf", "manual_03.pdf"];

const $ = (id) => document.getElementById(id);
const status = (msg) => ($("status").textContent = msg);

let embed, qa, retriever;
let chunks = [], vectors = [];

async function readPdf(data) {
  const doc = await pdfjs.getDocument({ data }).promise;
  const pages = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const tc = await (await doc.getPage(p)).getTextContent();
    pages.push(tc.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join(""));
  }
  return pages;
}

async function embedTexts(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const t = await embed(texts.slice(i, i + 16), { pooling: "mean", normalize: true });
    out.push(...t.tolist().map((v) => Float32Array.from(v)));
  }
  return out;
}

async function addFiles(files) {
  for (const { name, data } of files) {
    status(`Reading ${name}...`);
    const fresh = chunkPages(name, await readPdf(data));
    status(`Embedding ${name} (${fresh.length} chunks)...`);
    const vecs = await embedTexts(fresh.map((c) => c.text));
    const keep = chunks.map((c, i) => (c.file === name ? -1 : i)).filter((i) => i >= 0);
    chunks = keep.map((i) => chunks[i]).concat(fresh);
    vectors = keep.map((i) => vectors[i]).concat(vecs);
  }
  retriever = new Retriever(chunks, vectors);
  const docs = [...new Set(chunks.map((c) => c.file))];
  $("docs").textContent = docs.length ? docs.join(", ") : "none";
  status(`Ready: ${docs.length} PDFs, ${chunks.length} chunks.`);
}

async function loadUrls(urls) {
  const files = [];
  for (const u of urls) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
    files.push({ name: u.split("/").pop(), data: new Uint8Array(await r.arrayBuffer()) });
  }
  await addFiles(files);
}

function reset() {
  chunks = [];
  vectors = [];
  retriever = new Retriever([], []);
}

async function ask(question) {
  const [qvec] = await embedTexts([question]);
  return answer(question, retriever, qvec, (q, c) => qa(q, c));
}

function show(res) {
  const box = $("answer");
  box.hidden = false;
  box.className = res.found ? "panel answer" : "panel answer refused";
  $("answer-text").textContent = res.found ? res.answer : NOT_FOUND;
  $("answer-cite").textContent = res.found ? `${res.file} p.${res.page}` : "";
  $("answer-sentence").textContent = res.found ? `"${res.sentence}"` : "No passage matched the names and words in the question closely enough.";
  $("sources").innerHTML = "";
  for (const h of res.hits.slice(0, 3)) {
    const li = document.createElement("li");
    const b = document.createElement("b");
    b.textContent = `${h.chunk.file} p.${h.chunk.page}`;
    li.append(b, ` ${h.chunk.text.slice(0, 220)}${h.chunk.text.length > 220 ? "..." : ""}`);
    $("sources").append(li);
  }
}

async function main() {
  reset();
  status("Downloading the two small models (about 90 MB, cached after the first visit)...");
  [embed, qa] = await Promise.all([
    pipeline("feature-extraction", EMBED_MODEL, { dtype: "q8" }),
    pipeline("question-answering", QA_MODEL, { dtype: "q8" }),
  ]);
  window.pdfqa = { loadUrls, ask, reset, addFiles };
  if (!new URLSearchParams(location.search).has("bench")) await loadUrls(SAMPLES.map((s) => SAMPLE_BASE + s));
  $("ask").disabled = false;
  document.body.dataset.ready = "1";

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("q").value.trim();
    if (!q) return;
    $("ask").disabled = true;
    status("Thinking...");
    const t0 = performance.now();
    try {
      show(await ask(q));
      status(`Answered in ${((performance.now() - t0) / 1000).toFixed(1)} s, all in this browser.`);
    } finally {
      $("ask").disabled = false;
    }
  });
  document.querySelectorAll("[data-q]").forEach((b) =>
    b.addEventListener("click", () => {
      $("q").value = b.dataset.q;
      $("form").requestSubmit();
    }),
  );
  $("files").addEventListener("change", async (e) => {
    const files = await Promise.all(
      [...e.target.files].map(async (f) => ({ name: f.name, data: new Uint8Array(await f.arrayBuffer()) })),
    );
    if ($("replace").checked) reset();
    await addFiles(files);
  });
}

main().catch((err) => {
  status(`Could not start: ${err.message}`);
  document.body.dataset.ready = "error";
  console.error(err);
});
