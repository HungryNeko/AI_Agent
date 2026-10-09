import test from "node:test";
import assert from "node:assert/strict";
import { currentOperation, toolPreviewText } from "../src/operationPreview.js";

test("keeps current thinking and every tool from the same step", () => {
  const first = { type: "tool_call", step_id: 1, tool: "webSearch", text: "Search docs" };
  const second = { type: "tool_call", step_id: 1, tool: "mcp", text: "Read data" };
  assert.deepEqual(currentOperation([
    { type: "assistant_progress", step_id: 1, text: "I will search and query." }, first, second,
    { type: "parameters_changed", text: "Metadata changed" },
  ]), { thinking: "I will search and query.", tools: [first, second], attention: null });
});

test("new thinking immediately clears previous tools", () => {
  assert.deepEqual(currentOperation([
    { type: "assistant_progress", step_id: 1, text: "Old thinking" },
    { type: "tool_call", step_id: 1, tool: "webSearch" },
    { type: "assistant_progress", step_id: 2, text: "Now plot." },
  ]), { thinking: "Now plot.", tools: [], attention: null });
});

test("a tool-only step clears previous thinking and tools", () => {
  const tool = { type: "tool_call", step_id: 2, tool: "python" };
  assert.deepEqual(currentOperation([
    { type: "assistant_progress", step_id: 1, text: "Search" },
    { type: "tool_call", step_id: 1, tool: "webSearch" }, tool,
  ]), { thinking: "", tools: [tool], attention: null });
});

test("supports old history without step IDs and retains errors", () => {
  const error = { type: "error", text: "Request failed" };
  const tool = { type: "tool_call", tool: "curl" };
  const events = [{ type: "assistant_progress", text: "Check data" }, tool, error];
  assert.deepEqual(currentOperation(events), { thinking: "Check data", tools: [tool], attention: error });
  assert.deepEqual(currentOperation([...events, { type: "assistant_progress", text: "Use another source" }]),
    { thinking: "Use another source", tools: [], attention: null });
});

test("legacy tool-only steps start a new batch after result events", () => {
  const tool = { type: "tool_call", tool: "python" };
  assert.deepEqual(currentOperation([
    { type: "tool_call", tool: "mcp" }, { type: "parameters_changed" }, tool,
  ]), { thinking: "", tools: [tool], attention: null });
});

test("retains questions and approvals without replacing thinking", () => {
  for (const type of ["question_required", "approval_required", "ai_review"]) {
    const attention = { type, text: "Confirm" };
    assert.equal(currentOperation([{ type: "assistant_progress", text: "I need your confirmation" }, attention]).thinking, "I need your confirmation");
    assert.equal(currentOperation([attention]).attention, attention);
  }
});

test("tool previews preserve long URLs without dumping raw events", () => {
  assert.equal(toolPreviewText({ tool: "fileReader", path: "docs/report.md", data: [1, 2] }), "docs/report.md");
  const url = "https://example.com/" + "long-query-".repeat(40);
  assert.equal(toolPreviewText({ url }), url);
});
