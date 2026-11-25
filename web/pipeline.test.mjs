import { test } from "node:test";
import assert from "node:assert/strict";
import { chunkPages, hasPhrases, keyPhrases, tokenize } from "./pipeline.js";

test("tokenize drops stopwords and plural s", () => {
  assert.deepEqual(tokenize("What are the pumps of the Zinorin 400?"), ["pump", "zinorin", "400"]);
});

test("chunks never cross a page", () => {
  const chunks = chunkPages("a.pdf", ["One. ".repeat(300), "Two."]);
  assert.ok(chunks.length > 2);
  assert.deepEqual(chunks.at(-1), { file: "a.pdf", page: 2, text: "Two." });
});

test("key phrases need the exact name and number", () => {
  assert.deepEqual(keyPhrases("What is the maximum pressure of the Quorvex 900 unit?"), ["quorvex 900"]);
  assert.ok(!hasPhrases("The Quorvex 400 unit has 12 bar.", ["quorvex 900"]));
  assert.ok(hasPhrases("The Quorvex  900 unit has 12 bar.", ["quorvex 900"]));
});
