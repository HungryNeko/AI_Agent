const attentionTypes = new Set(["error", "question_required", "approval_required", "ai_review"]);

export function currentOperation(events) {
  let current = { thinking: "", tools: [], attention: null };
  let stepId;
  let legacyBatchClosed = false;

  for (const event of events) {
    if (event.type === "assistant_progress" || event.type === "tool_call") {
      const hasStep = event.step_id !== undefined && event.step_id !== null;
      const newStep = hasStep
        ? event.step_id !== stepId
        : event.type === "assistant_progress" || legacyBatchClosed;
      if (newStep) current = { thinking: "", tools: [], attention: null };
      stepId = event.step_id;
      legacyBatchClosed = false;
      if (event.type === "assistant_progress") current.thinking = event.text || "";
      else current.tools.push(event);
    } else {
      // Older saved conversations have no step IDs; results end their tool batch.
      legacyBatchClosed = current.tools.length > 0;
      if (attentionTypes.has(event.type)) current.attention = event;
    }
  }
  return current;
}

export function toolPreviewText(event) {
  return String(event.text || event.query || event.url || event.path || event.tool || "Tool").trim();
}
