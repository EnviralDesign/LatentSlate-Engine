import assert from "node:assert/strict";
import test from "node:test";
import { appendCollectionItem, moveCollectionItem, removeCollectionItem } from "../src/latentslate_engine/web/collections.js";

test("grouped row actions keep exact adapter references paired with their strengths", () => {
  const local = { source: "local", path: "M:\\Models\\LoRA A.safetensors" };
  const pinned = { source: "huggingface", repo: "models/lora", file: "adapter.safetensors", revision: "a".repeat(40), sha256: "b".repeat(64) };
  const artifacts = { key: "transformer_adapter_artifacts", mode: "fixed", value: [local, pinned] };
  const strengths = { key: "transformer_adapter_strengths", mode: "exposed", value: [0.25, 0.75], minimum: -1, maximum: 2 };
  const members = [
    { descriptor: { value_type: "artifact" }, field: artifacts },
    { descriptor: { value_type: "number" }, field: strengths },
  ];

  moveCollectionItem(members, 0, 1);
  assert.deepEqual(artifacts.value, [pinned, local]);
  assert.deepEqual(strengths.value, [0.75, 0.25]);
  moveCollectionItem(members, 1, -1);
  assert.deepEqual(artifacts.value, [local, pinned]);
  assert.deepEqual(strengths.value, [0.25, 0.75]);
  removeCollectionItem(members, 0);
  assert.deepEqual(artifacts.value, [pinned]);
  assert.deepEqual(strengths.value, [0.75]);
  appendCollectionItem(members);
  assert.deepEqual(artifacts.value, [pinned, { source: "local", path: "" }]);
  assert.deepEqual(strengths.value, [0.75, 1]);
  removeCollectionItem(members, 1);
  removeCollectionItem(members, 0);
  assert.deepEqual(artifacts.value, []);
  assert.deepEqual(strengths.value, []);
  appendCollectionItem(members);
  assert.deepEqual(artifacts.value, [{ source: "local", path: "" }]);
  assert.deepEqual(strengths, { key: "transformer_adapter_strengths", mode: "exposed", value: [1], minimum: -1, maximum: 2 });
  assert.equal(artifacts.mode, "fixed");
});

test("ordinary artifact and native adapter lists retain their item shapes", () => {
  for (const [value_type, expected] of [
    ["artifact", { source: "local", path: "" }],
    ["adapter", { artifact: { source: "local", path: "" }, strength: 1 }],
  ]) {
    const field = { value: [] };
    const members = [{ descriptor: { value_type }, field }];
    appendCollectionItem(members);
    appendCollectionItem(members);
    assert.deepEqual(field.value, [expected, expected]);
    assert.notEqual(field.value[0], field.value[1]);
    removeCollectionItem(members, 0);
    assert.deepEqual(field.value, [expected]);
  }
});
