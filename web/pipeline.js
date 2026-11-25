// Browser version of the pdfqa pipeline: page text -> chunks -> hybrid retrieval
// (BM25 + MiniLM cosine, reciprocal rank fusion) -> refusal checks -> extractive QA span.
// Mirrors src/pdfqa/text.py and src/pdfqa/retrieve.py so the two stay comparable.

export const NOT_FOUND = "Not in these documents.";

const STOPWORDS = new Set(
  ("a an the of to in on at for and or is are was were be been by with as it its this that " +
    "these those what which who whom how when where why does do did from has have had not can " +
    "will would should may about into than then there their they them you your we our i").split(" "),
);
const WORD = /[a-z0-9][a-z0-9\-.]*[a-z0-9]|[a-z0-9]/g;

export function tokenize(text) {
  const out = [];
  for (const raw of text.toLowerCase().match(WORD) ?? []) {
    let tok = raw.replace(/[.-]+$/, "");
    if (!tok || STOPWORDS.has(tok)) continue;
    if (tok.length > 3 && tok.endsWith("s") && !tok.endsWith("ss")) tok = tok.slice(0, -1);
    out.push(tok);
  }
  return out;
}

export function splitPage(text, maxChars = 700, overlap = 100) {
  text = text.replace(/\s+/g, " ").trim();
  if (!text) return [];
  if (text.length <= maxChars) return [text];
  const pieces = [];
  let cur = "";
  for (let sent of text.split(/(?<=[.!?])\s+/)) {
    while (sent.length > maxChars) {
      if (cur) pieces.push(cur), (cur = "");
      pieces.push(sent.slice(0, maxChars));
      sent = sent.slice(maxChars - overlap);
    }
    if (cur && cur.length + 1 + sent.length > maxChars) {
      pieces.push(cur);
      const tail = overlap ? cur.slice(-overlap) : "";
      cur = tail ? (tail.split(" ").slice(1).join(" ") + " " + sent).trim() : sent;
    } else {
      cur = `${cur} ${sent}`.trim();
    }
  }
  if (cur) pieces.push(cur);
  return pieces;
}

/** Chunks never cross a page, so every citation is a real page. */
export function chunkPages(file, pages) {
  const chunks = [];
  pages.forEach((pageText, i) => {
    for (const text of splitPage(pageText)) chunks.push({ file, page: i + 1, text });
  });
  return chunks;
}

/** Names and numbers in the question ("Zinorin 400", "2019"): the answer chunk must contain them. */
export function keyPhrases(question) {
  const words = question.replace(/[?!.,;:"]/g, " ").split(/\s+/).filter(Boolean);
  const phrases = [];
  let cur = [];
  words.forEach((w, i) => {
    const first = i === 0 && STOPWORDS.has(w.toLowerCase());
    if (!first && /^([A-Z][\w-]*|\d[\d.,-]*)$/.test(w)) cur.push(w);
    else if (cur.length) phrases.push(cur.join(" ")), (cur = []);
  });
  if (cur.length) phrases.push(cur.join(" "));
  return phrases.map((p) => p.toLowerCase());
}

const norm = (s) => s.toLowerCase().replace(/\s+/g, " ");
export const hasPhrases = (text, phrases) => phrases.every((p) => norm(text).includes(p));

const RRF_K = 60;

export class Retriever {
  constructor(chunks, vectors, opts = {}) {
    this.chunks = chunks;
    this.vectors = vectors; // array of Float32Array, unit length
    this.minCoverage = opts.minCoverage ?? 0.6;
    this.minPhrase = opts.minPhrase ?? 0.5;
    this.minCosine = opts.minCosine ?? 0.55;
    this.tokens = chunks.map((c) => tokenize(c.text));
    this.sets = this.tokens.map((t) => new Set(t));
    this.pairs = this.tokens.map((t) => new Set(t.slice(1).map((w, i) => `${t[i]} ${w}`)));
    this.tf = this.tokens.map((t) => t.reduce((m, w) => m.set(w, (m.get(w) ?? 0) + 1), new Map()));
    const n = Math.max(chunks.length, 1);
    const df = new Map();
    for (const s of this.sets) for (const w of s) df.set(w, (df.get(w) ?? 0) + 1);
    this.idf = new Map([...df].map(([w, d]) => [w, Math.log(1 + (n - d + 0.5) / (d + 0.5))]));
    this.avgLen = this.tokens.reduce((a, t) => a + t.length, 0) / n || 1;
  }

  bm25(terms) {
    const k1 = 1.5, b = 0.75;
    return this.tf.map((tf, i) => {
      const len = this.tokens[i].length;
      let s = 0;
      for (const t of terms) {
        const f = tf.get(t) ?? 0;
        if (f) s += (this.idf.get(t) * f * (k1 + 1)) / (f + k1 * (1 - b + (b * len) / this.avgLen));
      }
      return s;
    });
  }

  search(query, qvec, k = 5) {
    if (!this.chunks.length) return [];
    const ordered = tokenize(query);
    const qPairs = new Set(ordered.slice(1).map((w, i) => `${ordered[i]} ${w}`));
    const terms = [...new Set(ordered)];
    const cos = this.vectors.map((v) => v.reduce((a, x, j) => a + x * qvec[j], 0));
    const bm = this.bm25(terms);
    const pool = Math.min(Math.max(k * 4, 20), this.chunks.length);
    const byDesc = (arr) => arr.map((_, i) => i).sort((a, b) => arr[b] - arr[a]).slice(0, pool);
    const fused = new Map();
    const add = (list) => list.forEach((i, r) => fused.set(i, (fused.get(i) ?? 0) + 1 / (RRF_K + r + 1)));
    add(byDesc(cos));
    add(byDesc(bm).filter((i) => bm[i] > 0));
    return [...fused.keys()]
      .sort((a, b) => fused.get(b) - fused.get(a))
      .slice(0, k)
      .map((i) => {
        const cov = terms.length ? terms.filter((t) => this.sets[i].has(t)).length / terms.length : 0;
        const phrase = qPairs.size ? [...qPairs].filter((p) => this.pairs[i].has(p)).length / qPairs.size : cov;
        return { chunk: this.chunks[i], score: fused.get(i), cosine: cos[i], bm25: bm[i], coverage: cov, phrase };
      });
  }

  strong(hit) {
    return (hit.coverage >= this.minCoverage && hit.phrase >= this.minPhrase) || hit.cosine >= this.minCosine;
  }
}

/**
 * Answer from the retrieved chunks with an extractive QA model.
 * qa(question, context) -> {answer, score}. Returns {found, answer, file, page, score, sentence, hits}.
 */
export async function answer(question, retriever, qvec, qa, opts = {}) {
  const k = opts.k ?? 5;
  const minQa = opts.minQa ?? 0.2;
  const hits = retriever.search(question, qvec, k);
  const phrases = keyPhrases(question);
  const candidates = hits.slice(0, 3).filter((h) => retriever.strong(h) && hasPhrases(h.chunk.text, phrases));
  let best = null;
  for (const h of candidates) {
    const out = await qa(question, h.chunk.text);
    const r = Array.isArray(out) ? out[0] : out;
    if (r?.answer?.trim() && (!best || r.score > best.score)) best = { ...r, hit: h };
  }
  if (!best || best.score < minQa) return { found: false, answer: NOT_FOUND, hits, score: best?.score ?? 0 };
  const sentence =
    best.hit.chunk.text.split(/(?<=[.!?])\s+/).find((s) => s.includes(best.answer.trim())) ?? best.answer;
  return {
    found: true,
    answer: best.answer.trim(),
    sentence,
    file: best.hit.chunk.file,
    page: best.hit.chunk.page,
    score: best.score,
    hits,
  };
}
