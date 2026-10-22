import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  Archive,
  ArrowUp,
  AtSign,
  Bot,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Code2,
  Copy,
  Database,
  Download,
  FileText,
  GitBranch,
  Globe,
  GripVertical,
  Image,
  Key,
  Languages,
  ListChecks,
  ListOrdered,
  Maximize2,
  MessageSquare,
  Minus,
  Paperclip,
  Pause,
  Pencil,
  Play,
  Plug,
  Plus,
  RefreshCw,
  Scissors,
  Save,
  Send,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Sun,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import "katex/dist/katex.min.css";
import "./styles.css";
import { currentOperation, toolPreviewText } from "./operationPreview.js";
import { MenuItem, Pill, PopPill, Resizer, Segmented, StepSlider, splitColumns, usePane } from "./ui.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "";
// Events fetched per page; long conversations load newest-first and fetch older pages on scroll-up.
const EVENT_PAGE = 60;

function createRunId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const emptyOptions = {
  model: "",
  system_prompt: "",
  web_search_mode: "auto",
  web_search_provider: "duckduckgo",
  web_search_auto_switch: true,
  rag_mode: "auto",
  rag_include_knowledge: true,
  rag_include_memory: true,
  rag_include_skills: true,
  curl_mode: "auto",
  python_mode: "auto",
  file_reader_mode: "auto",
  file_editor_mode: "auto",
  file_editor_approval: "auto",
  mcp_mode: "auto",
  history_mode: "auto",
  automation_mode: "auto",
  question_mode: "light",
  developer_mode: false,
  conversation_mode: "agent",
  max_tool_rounds: 20,
  subagent_model: "",
  reasoning_effort: "default",
  subagent_reasoning_effort: "default",
};

const TEXT = {
  chat: ["对话", "Chat"],
  data: ["数据", "Data"],
  config: ["模型", "Models"],
  mcp: ["MCP", "MCP"],
  tools: ["工具", "Tools"],
  settings: ["设置", "Settings"],
  automation: ["自动化", "Automation"],
  history: ["历史", "History"],
  newChat: ["新对话", "New Chat"],
  compress: ["压缩", "Compress"],
  send: ["发送", "Send"],
  running: ["运行中", "Running"],
  stop: ["\u505c\u6b62", "Stop"],
  uploadFile: ["批量上传文件", "Upload files"],
  uploadImage: ["批量上传图片", "Upload images"],
  system: ["系统", "System"],
};

function App() {
  const [tab, setTab] = useState("chat");
  const [visited, setVisited] = useState({ chat: true });
  const [models, setModels] = useState([]);
  const [nativeWebProviders, setNativeWebProviders] = useState([]);
  const [options, setOptionsState] = useState(emptyOptions);
  const [theme, setThemeState] = useState("system");
  const [language, setLanguageState] = useState("zh");
  const pendingSettingsPatchRef = useRef({});
  const settingsSaveTimerRef = useRef(null);
  const label = useLabel(language);
  const text = useText(language);

  useEffect(() => {
    setVisited((current) => (current[tab] ? current : { ...current, [tab]: true }));
  }, [tab]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    reloadModels().catch(() => setModels([]));
  }, []);

  useEffect(() => {
    reloadSettings().catch(() => {});
  }, []);

  async function reloadSettings() {
    const data = await fetchJson("/api/settings");
    const saved = data.settings || {};
    const savedUi = saved.ui || {};
    const savedChat = saved.chat || {};
    setThemeState(savedUi.theme || "system");
    setLanguageState(savedUi.language || "zh");
    setOptionsState((current) => ({
      ...emptyOptions,
      ...savedChat,
      model: savedChat.model || current.model || "",
    }));
    return saved;
  }

  async function reloadModels() {
    const data = await fetchJson("/api/models");
    setModels(data.models || []);
    setNativeWebProviders(data.nativeWebProviders || []);
    setOptionsState((current) => ({ ...current, model: current.model || data.defaultModel || "" }));
    return data;
  }

  function scheduleSettingsPatch(patch) {
    pendingSettingsPatchRef.current = deepMerge(pendingSettingsPatchRef.current, patch);
    window.clearTimeout(settingsSaveTimerRef.current);
    settingsSaveTimerRef.current = window.setTimeout(() => {
      const nextPatch = pendingSettingsPatchRef.current;
      pendingSettingsPatchRef.current = {};
      fetchJson("/api/settings", { method: "PATCH", body: { patch: nextPatch } }).catch(() => {});
    }, 250);
  }

  function setOptions(nextOptions) {
    setOptionsState((current) => {
      const next = typeof nextOptions === "function" ? nextOptions(current) : nextOptions;
      scheduleSettingsPatch({ chat: next });
      return next;
    });
  }

  function setTheme(value) {
    setThemeState(value);
    scheduleSettingsPatch({ ui: { theme: value } });
  }

  function setLanguage(value) {
    setLanguageState(value);
    scheduleSettingsPatch({ ui: { language: value } });
  }

  return (
    <main className="appShell">
      <header className="topbar">
        <div>
          <h1>AI AGENT</h1>
          <p>对话、工具、RAG、技能、记忆、知识和 MCP</p>
        </div>
        <nav className="tabs" aria-label="Main views">
          <TabButton active={tab === "chat"} onClick={() => setTab("chat")} icon={<Send size={16} />} label={label("chat")} />
          <TabButton active={tab === "data"} onClick={() => setTab("data")} icon={<Database size={16} />} label={label("data")} />
          <TabButton active={tab === "config"} onClick={() => setTab("config")} icon={<Settings size={16} />} label={label("config")} />
          <TabButton active={tab === "automation"} onClick={() => setTab("automation")} icon={<RefreshCw size={16} />} label={label("automation")} />
          <TabButton active={tab === "mcp"} onClick={() => setTab("mcp")} icon={<Plug size={16} />} label={label("mcp")} />
          <TabButton active={tab === "tools"} onClick={() => setTab("tools")} icon={<Wrench size={16} />} label={label("tools")} />
          <TabButton active={tab === "system"} onClick={() => setTab("system")} icon={<Sun size={16} />} label={label("system")} />
        </nav>
      </header>

      <TabPane id="chat" tab={tab} visited={visited}>
        <ChatView
          models={models}
          nativeWebProviders={nativeWebProviders}
          options={options}
          setOptions={setOptions}
          label={label}
          text={text}
          onSettingsChanged={reloadSettings}
          onModelsChanged={reloadModels}
        />
      </TabPane>
      <TabPane id="data" tab={tab} visited={visited}><DataView text={text} models={models} /></TabPane>
      <TabPane id="config" tab={tab} visited={visited}><ConfigView text={text} onSaved={reloadModels} /></TabPane>
      <TabPane id="automation" tab={tab} visited={visited}><AutomationView options={options} setOptions={setOptions} text={text} /></TabPane>
      <TabPane id="mcp" tab={tab} visited={visited}><McpView text={text} /></TabPane>
      <TabPane id="tools" tab={tab} visited={visited}><CustomToolsView text={text} /></TabPane>
      <TabPane id="system" tab={tab} visited={visited}><SystemView theme={theme} setTheme={setTheme} language={language} setLanguage={setLanguage} text={text} /></TabPane>
    </main>
  );
}

// Keep visited tabs mounted (hidden) so state, scroll position and in-flight
// streams survive tab switches instead of reloading everything.
function TabPane({ id, tab, visited, children }) {
  if (!visited[id]) return null;
  return (
    <div className="tabPane" hidden={tab !== id}>
      {children}
    </div>
  );
}

function TabButton({ active, onClick, icon, label }) {
  return (
    <button className={active ? "tab active" : "tab"} onClick={onClick} type="button">
      {icon}
      <span>{label}</span>
    </button>
  );
}

function ChatView({ models, nativeWebProviders = [], options, setOptions, label, text, onSettingsChanged, onModelsChanged }) {
  const [message, setMessage] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [events, setEvents] = useState([]);
  const [state, setStateRaw] = useState(null);
  const [livePlan, setLivePlan] = useState(null);
  const [selectMode, setSelectMode] = useState(false);
  // Inline rename / delete confirmation: window.prompt and confirm are blocked in embedded browsers.
  const [renamingId, setRenamingId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState("");
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);
  const [eventOffset, setEventOffset] = useState(0);
  const [hasMoreEvents, setHasMoreEvents] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  // False while local event indices may not match the saved conversation (until the next resync).
  const [indexTrusted, setIndexTrusted] = useState(true);
  const eventOffsetRef = useRef(0);
  // Background sub-agents of this conversation and the server-started "wake" turn (if one is running).
  const [subagents, setSubagents] = useState({});
  const [agentTasks, setAgentTasks] = useState([]);
  const conversationIdRef = useRef("");
  const wakeRunIdRef = useRef("");
  const liveHandlerRef = useRef(() => {});
  const busyRef = useRef(false);
  const loadingOlderRef = useRef(false);
  const scrollRestoreRef = useRef(null);
  const [conversationId, setConversationId] = useState("");
  const [conversationModel, setConversationModel] = useState("");
  const [conversations, setConversations] = useState([]);
  const [historyStatus, setHistoryStatus] = useState("");
  const historyPane = usePane("chat-history", { size: 280, min: 200, max: 480, side: "start" });
  const [security, setSecurity] = useState(null);
  const [securityError, setSecurityError] = useState("");
  const [mentionOptions, setMentionOptions] = useState([]);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionOpen, setMentionOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [inputMode, setInputMode] = useState("queue");
  const [queuedMessages, setQueuedMessages] = useState([]);
  const elapsed = useElapsed(busy);
  const outputRef = useRef(null);
  const composerRef = useRef(null);
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const atBottomRef = useRef(true);
  const abortRef = useRef(null);
  const activeRunIdRef = useRef("");
  const stopRequestedRef = useRef(false);

  useEffect(() => {
    refreshConversations();
    fetchJson("/api/security").then(setSecurity).catch(() => {});
  }, []);

  async function changeSecurity(level) {
    setSecurityError("");
    try {
      setSecurity(await fetchJson("/api/security/level", { method: "PUT", body: { level } }));
    } catch (error) {
      setSecurityError(errorDetail(error));
    }
  }

  // Authoritative state replaces any plan streamed ahead of the final event.
  function setState(next) {
    setLivePlan(null);
    setStateRaw(next);
  }

  // The plan card is pinned below the transcript. While a turn runs, hide it so a
  // tall stale card cannot cover the new messages; it reappears (and scrolls into
  // view) once the turn ends.
  eventOffsetRef.current = eventOffset;
  conversationIdRef.current = conversationId;
  liveHandlerRef.current = handleLiveEvent;

  const displayPlan = livePlan || state?.plan || null;

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  // Things that happen outside this tab's own request: sub-agent updates and server-started wake turns.
  useEffect(() => {
    setSubagents({});
    setAgentTasks([]);
    if (!conversationId) return undefined;
    let closed = false;
    const controller = new AbortController();
    (async () => {
      let attempt = 0;
      while (!closed) {
        try {
          const response = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}/live`, { signal: controller.signal });
          if (response.ok && response.body) {
            attempt = 0;
            await readSse(response.body, (liveEvent) => liveHandlerRef.current(liveEvent));
          }
        } catch (error) {
          if (closed || error?.name === "AbortError") return;
        }
        if (closed) return;
        attempt += 1;
        await new Promise((resolve) => window.setTimeout(resolve, Math.min(15000, 1000 * attempt)));
      }
    })();
    return () => {
      closed = true;
      controller.abort();
    };
  }, [conversationId]);

  // Keep the viewport anchored when older events are prepended above it.
  useLayoutEffect(() => {
    const restore = scrollRestoreRef.current;
    const node = outputRef.current;
    if (!restore || !node) return;
    scrollRestoreRef.current = null;
    node.scrollTop = restore.top + (node.scrollHeight - restore.height);
  }, [events]);

  // A short page may not fill the viewport; keep pulling older pages until it scrolls.
  useEffect(() => {
    const node = outputRef.current;
    if (hasMoreEvents && !loadingOlder && !busy && node && node.scrollHeight <= node.clientHeight + 40) loadOlderEvents();
  }, [events, hasMoreEvents, loadingOlder, busy]);

  useEffect(() => {
    if (atBottomRef.current) outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight });
  }, [events, busy, displayPlan?.content, displayPlan?.status]);

  useEffect(() => {
    if (busy || pendingQuestion || queuedMessages.length === 0) return;
    const [next, ...rest] = queuedMessages;
    setQueuedMessages(rest);
    runChatTurn(next.message, next.attachments, next.displayMessage);
  }, [busy, pendingQuestion, queuedMessages]);

  function trackScroll() {
    const node = outputRef.current;
    if (!node) return;
    atBottomRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
    if (node.scrollTop < 200) loadOlderEvents();
  }

  // Show a conversation (or one page of it), newest events last.
  function applyConversation(data, { toBottom = true } = {}) {
    const all = data.events || [];
    const paged = data.event_offset !== undefined;
    const shown = paged ? all : all.slice(-EVENT_PAGE);
    const offset = paged ? data.event_offset : all.length - shown.length;
    if (toBottom) atBottomRef.current = true;
    setEvents(shown);
    setEventOffset(offset);
    setHasMoreEvents(paged ? !!data.has_more : offset > 0);
    setIndexTrusted(true);
  }

  async function loadOlderEvents() {
    if (!conversationId || !hasMoreEvents || loadingOlderRef.current || busyRef.current || eventOffsetRef.current <= 0) return;
    loadingOlderRef.current = true;
    setLoadingOlder(true);
    const node = outputRef.current;
    const anchor = { height: node?.scrollHeight || 0, top: node?.scrollTop || 0 };
    try {
      const data = await fetchJson(
        `/api/conversations/${encodeURIComponent(conversationId)}?limit=${EVENT_PAGE}&before=${eventOffsetRef.current}&state=false`,
      );
      scrollRestoreRef.current = anchor;
      setEvents((items) => [...(data.events || []), ...items]);
      setEventOffset(data.event_offset || 0);
      setHasMoreEvents(!!data.has_more);
    } catch {
      setHasMoreEvents(false);
    } finally {
      loadingOlderRef.current = false;
      setLoadingOlder(false);
    }
  }

  // Replace the local tail with what the server saved so indices stay exact.
  async function resyncEvents(id, { force = false } = {}) {
    if (!id || (busyRef.current && !force)) return;
    try {
      const data = await fetchJson(`/api/conversations/${encodeURIComponent(id)}?since=${eventOffsetRef.current}&state=false`);
      setEvents(data.events || []);
      setEventOffset(data.event_offset || 0);
      setHasMoreEvents(!!data.has_more);
      setIndexTrusted(true);
    } catch {
      /* keep the local view; actions stay disabled until the next successful sync */
    }
  }

  async function refreshConversations() {
    try {
      const data = await fetchJson("/api/conversations");
      setConversations(data.conversations || []);
      setMentionOptions((current) => mergeMentionOptions(current, conversationMentionOptions(data.conversations || [], text)));
    } catch (error) {
      setHistoryStatus(`历史加载失败: ${String(error.message || error)}`);
    }
  }

  async function openConversation(id) {
    if (!id || busy) return;
    const data = await fetchJson(`/api/conversations/${encodeURIComponent(id)}?limit=${EVENT_PAGE}`);
    setConversationId(data.id || id);
    applyConversation(data);
    setState(data.state || null);
    setConversationModel(data.state?.model || "");
    setPendingQuestion(data.state?.question_pending || null);
    setAttachments([]);
    setHistoryStatus("");
  }

  function startRename(item) {
    setConfirmDeleteId("");
    setRenamingId(item.id);
    setRenameDraft(item.title || "");
  }

  async function commitRename() {
    const id = renamingId;
    const title = renameDraft.trim();
    setRenamingId("");
    if (!id || !title) return;
    try {
      await fetchJson(`/api/conversations/${encodeURIComponent(id)}`, { method: "PATCH", body: { title } });
      await refreshConversations();
    } catch (error) {
      setHistoryStatus(`${text("重命名失败：", "Rename failed: ")}${String(error.message || error)}`);
    }
  }

  async function deleteConversation(id) {
    setConfirmDeleteId("");
    try {
      await fetchJson(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (conversationId === id) newConversation();
      await refreshConversations();
    } catch (error) {
      setHistoryStatus(`${text("删除失败：", "Delete failed: ")}${String(error.message || error)}`);
    }
  }

  function toggleSelectMode() {
    setSelectMode((on) => !on);
    setSelectedIds([]);
    setConfirmBulk(false);
    setConfirmDeleteId("");
    setRenamingId("");
  }

  function toggleSelected(id) {
    setConfirmBulk(false);
    setSelectedIds((ids) => (ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id]));
  }

  const allSelected = conversations.length > 0 && selectedIds.length === conversations.length;

  function toggleSelectAll() {
    setConfirmBulk(false);
    setSelectedIds(allSelected ? [] : conversations.map((item) => item.id));
  }

  async function deleteSelectedConversations() {
    // The conversation being streamed into cannot disappear mid-turn.
    const ids = selectedIds.filter((id) => !(busy && id === conversationId));
    if (ids.length === 0) return;
    setConfirmBulk(false);
    const results = await Promise.allSettled(
      ids.map((id) => fetchJson(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" })),
    );
    const failed = results.filter((result) => result.status === "rejected").length;
    if (ids.includes(conversationId) && !busy) newConversation();
    setSelectedIds([]);
    setSelectMode(false);
    setHistoryStatus(
      failed
        ? text(`已删除 ${ids.length - failed} 条，${failed} 条失败`, `Deleted ${ids.length - failed}, ${failed} failed`)
        : text(`已删除 ${ids.length} 条对话`, `Deleted ${ids.length} conversations`),
    );
    await refreshConversations();
  }

  function newConversation() {
    if (busy) return;
    setConversationId("");
    setEvents([]);
    setEventOffset(0);
    setHasMoreEvents(false);
    setIndexTrusted(true);
    setState(null);
    setConversationModel("");
    setAttachments([]);
    setPreviewImage(null);
    setPendingQuestion(null);
    setQueuedMessages([]);
    setPaused(false);
    setHistoryStatus("");
  }

  async function compressConversation() {
    if (!conversationId || busy) return;
    try {
      const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/compress`, { method: "POST" });
      setState(data.state || null);
      setHistoryStatus("已压缩当前上下文；完整 JSON 历史仍然保留。");
      await refreshConversations();
    } catch (error) {
      setHistoryStatus(`压缩失败: ${String(error.message || error)}`);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    if (!message.trim() && attachments.length === 0) return;
    const nextMessage = message.trim() || "请读取这些附件。";
    const nextAttachments = attachments;
    setMessage("");
    setAttachments([]);
    setMentionOpen(false);
    if (busy) {
      const item = { id: createRunId(), message: nextMessage, attachments: nextAttachments, displayMessage: nextMessage };
      setQueuedMessages((items) => inputMode === "insert" ? [item, ...items] : [...items, item]);
      if (inputMode === "insert") stopOutput();
      return;
    }
    await runChatTurn(nextMessage, nextAttachments, nextMessage);
  }

  // Apply one event of a running turn to the view (shared by user turns and server-started wake turns).
  function applyStreamEvent(eventData) {
    setEvents((items) => [...items, eventData]);
    if (eventData.type === "plan_updated" || eventData.type === "plan_ready") {
      // Show the plan as soon as the tool call returns, even if the turn never reaches a final state.
      atBottomRef.current = true;
      setLivePlan({ name: eventData.name, content: eventData.content, status: eventData.status });
    }
    if (eventData.type === "question_required" || eventData.type === "approval_request") setPendingQuestion(eventData);
    if (eventData.type === "model_changed") setConversationModel(eventData.text || "");
    if (eventData.type === "models_changed") onModelsChanged?.().catch(() => {});
    if (eventData.type === "assistant" && eventData.state) {
      setState(eventData.state);
      setConversationModel(eventData.state.model || conversationModel || "");
      setPendingQuestion(eventData.state.question_pending || null);
    }
    if (eventData.type === "settings_changed") onSettingsChanged?.().catch(() => {});
  }

  function handleLiveEvent(liveEvent) {
    if (liveEvent.conversation_id && liveEvent.conversation_id !== conversationIdRef.current) return;
    switch (liveEvent.type) {
      case "subagents_snapshot":
        setSubagents(Object.fromEntries((liveEvent.agents || []).map((agent) => [agent.id, agent])));
        setAgentTasks(liveEvent.tasks || []);
        return;
      case "tasklist_update":
        setAgentTasks(liveEvent.tasks || []);
        return;
      case "subagent_update":
        if (liveEvent.agent) setSubagents((current) => ({ ...current, [liveEvent.agent.id]: liveEvent.agent }));
        return;
      case "wake_started":
        wakeRunIdRef.current = liveEvent.run_id;
        activeRunIdRef.current = liveEvent.run_id;
        atBottomRef.current = true;
        busyRef.current = true;
        setBusy(true);
        return;
      case "wake_finished":
        if (wakeRunIdRef.current === liveEvent.run_id) {
          wakeRunIdRef.current = "";
          if (activeRunIdRef.current === liveEvent.run_id) activeRunIdRef.current = "";
          busyRef.current = false;
          setBusy(false);
          setPaused(false);
          resyncEvents(conversationIdRef.current, { force: true });
          refreshConversations();
        }
        return;
      default:
        if (liveEvent.run_id && liveEvent.run_id === wakeRunIdRef.current) applyStreamEvent(liveEvent);
    }
  }

  async function cancelSubagent(agentId) {
    try {
      await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/subagents/${encodeURIComponent(agentId)}/cancel`, { method: "POST" });
    } catch (error) {
      setHistoryStatus(`${text("取消失败：", "Cancel failed: ")}${String(error.message || error)}`);
    }
  }

  async function runChatTurn(nextMessage, nextAttachments = [], displayMessage = nextMessage, overrides = {}) {
    if (!nextMessage.trim() || busy) return;
    atBottomRef.current = true;
    setBusy(true);
    busyRef.current = true;
    const wasNewConversation = !conversationId;
    let activeConversationId = conversationId;
    let localFailure = false;
    const runId = createRunId();
    const controller = new AbortController();
    activeRunIdRef.current = runId;
    stopRequestedRef.current = false;
    abortRef.current = controller;
    setEvents((items) => [...items, { type: "user", text: displayMessage, attachments: nextAttachments }]);

    try {
      let streamCompleted = false;
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: nextMessage,
          display_message: displayMessage,
          run_id: runId,
          attachments: nextAttachments,
          conversation_id: conversationId || null,
          state: overrides.state === undefined ? state : overrides.state,
          options: normalizeOptions({
            ...options,
            ...(overrides.options || {}),
            model: conversationModel || overrides.options?.model || options.model,
          }),
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      if (!response.body) throw new Error("HTTP response has no stream body");
      await readSse(response.body, (eventData) => {
        if (["assistant", "stopped"].includes(eventData.type) || (eventData.type === "error" && eventData.terminal)) {
          streamCompleted = true;
        }
        if (eventData.conversation_id) {
          activeConversationId = eventData.conversation_id;
          setConversationId(eventData.conversation_id);
        }
        applyStreamEvent(eventData);
      });
      if (!streamCompleted) {
        throw new Error(text("响应流意外结束，后端没有返回最终结果。", "The response stream ended without a final result."));
      }
      await refreshConversations();
      await resyncEvents(activeConversationId, { force: true });
      if (wasNewConversation) {
        // The model-written title lands a moment after the turn is saved.
        window.setTimeout(() => refreshConversations(), 2500);
        window.setTimeout(() => refreshConversations(), 7000);
      }
    } catch (error) {
      if (!stopRequestedRef.current && error?.name !== "AbortError") {
        localFailure = true;
        setEvents((items) => [...items, { type: "error", terminal: true, text: String(error.message || error) }]);
      }
      if (stopRequestedRef.current || error?.name === "AbortError" || localFailure) setIndexTrusted(false);
    } finally {
      if (activeRunIdRef.current === runId) {
        abortRef.current = null;
        activeRunIdRef.current = "";
        stopRequestedRef.current = false;
        if (!wakeRunIdRef.current) setBusy(false);
        setPaused(false);
      }
    }
  }

  function stopOutput() {
    if (!busy) return;
    const runId = activeRunIdRef.current;
    stopRequestedRef.current = true;
    if (runId) {
      fetchJson("/api/chat/stop", { method: "POST", body: { run_id: runId } }).catch(() => {});
    }
    abortRef.current?.abort();
    setEvents((items) => {
      if (runId && items.some((item) => item.type === "stopped" && item.run_id === runId)) return items;
      return [...items, { type: "stopped", text: "AI output stopped.", run_id: runId, conversation_id: conversationId || undefined }];
    });
    setBusy(false);
    setPaused(false);
    setIndexTrusted(false);
    window.setTimeout(() => refreshConversations().catch(() => {}), 500);
    // Once the server has saved the interrupted turn, line the local list up with it again.
    if (conversationId) window.setTimeout(() => resyncEvents(conversationId), 1200);
  }

  async function loadMentionOptions() {
    const base = toolMentionOptions(text);
    try {
      const [history, memory, skills, knowledge] = await Promise.all([
        fetchJson("/api/conversations").catch(() => ({ conversations: [] })),
        fetchJson("/api/data/files?kind=memory").catch(() => ({ items: [], files: [] })),
        fetchJson("/api/data/files?kind=skills").catch(() => ({ items: [], files: [] })),
        fetchJson("/api/data/files?kind=knowledge").catch(() => ({ items: [], files: [] })),
      ]);
      setMentionOptions([
        ...base,
        ...conversationMentionOptions(history.conversations || [], text),
        ...fileMentionOptions("memory", memory.items || memory.files || [], text),
        ...fileMentionOptions("skill", skills.items || skills.files || [], text),
        ...fileMentionOptions("knowledge", knowledge.items || knowledge.files || [], text),
      ]);
    } catch {
      setMentionOptions(base);
    }
  }

  function updateMessage(value, selectionStart) {
    setMessage(value);
    const beforeCursor = value.slice(0, selectionStart ?? value.length);
    const match = beforeCursor.match(/@([^\s@]*)$/);
    if (match) {
      setMentionQuery(match[1].toLowerCase());
      setMentionOpen(true);
      if (mentionOptions.length === 0) loadMentionOptions();
    } else {
      setMentionOpen(false);
    }
  }

  function insertMention(option) {
    const textarea = composerRef.current;
    const cursor = textarea?.selectionStart ?? message.length;
    const before = message.slice(0, cursor).replace(/@([^\s@]*)$/, option.token);
    const next = `${before} ${message.slice(cursor)}`;
    setMessage(next);
    setMentionOpen(false);
    requestAnimationFrame(() => composerRef.current?.focus());
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function uploadSelectedFile(event) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length) return;
    setHistoryStatus(files.length > 1 ? `上传中: ${files.length} 个文件` : "上传中");
    const results = await Promise.allSettled(files.map(uploadOneFile));
    const uploaded = results.filter((result) => result.status === "fulfilled").map((result) => result.value);
    const failed = results.filter((result) => result.status === "rejected");
    if (uploaded.length) {
      setAttachments((current) => [...current, ...uploaded]);
      const tokens = uploaded.map((item) => `@file:${item.path}`).join(" ");
      setMessage((current) => `${current}${current ? " " : ""}${tokens}`);
    }
    if (failed.length) {
      setHistoryStatus(`上传失败: ${failed.length}/${files.length}`);
    } else {
      setHistoryStatus(files.length > 1 ? `已上传 ${uploaded.length} 个文件` : "");
    }
  }

  async function togglePause() {
    const runId = activeRunIdRef.current;
    if (!busy || !runId) return;
    const nextPaused = !paused;
    await fetchJson(`/api/chat/${nextPaused ? "pause" : "resume"}`, { method: "POST", body: { run_id: runId } });
    setPaused(nextPaused);
  }

  // Fork into a brand-new conversation. Forking at a user message stops right before it
  // and puts its text back in the composer, so it can be sent (or changed) as a new path.
  async function branchConversation(event) {
    if (!conversationId || busy || !Number.isInteger(event?.eventIndex)) return;
    const atUserMessage = event.type === "user" && !event.source;
    try {
      const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/branch`, {
        method: "POST",
        body: { event_index: event.eventIndex, before: atUserMessage },
      });
      setConversationId(data.id);
      applyConversation(data);
      setState(data.state || null);
      setPendingQuestion(null);
      setConversationModel(data.state?.model || conversationModel || "");
      if (atUserMessage) {
        setMessage(event.text || "");
        requestAnimationFrame(() => composerRef.current?.focus());
      }
      setHistoryStatus(text("已创建新的对话分支", "Forked into a new conversation"));
      await refreshConversations();
    } catch (error) {
      setHistoryStatus(`${text("分叉失败：", "Fork failed: ")}${String(error.message || error)}`);
    }
  }

  // Rewrite a past user message: drop it and everything after it, then re-run from there.
  async function editUserMessage(event, newText) {
    const next = (newText || "").trim();
    if (!conversationId || busy || !next || !Number.isInteger(event?.eventIndex)) return;
    try {
      const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/truncate`, {
        method: "POST",
        body: { event_index: event.eventIndex },
      });
      setEvents((items) => items.slice(0, Math.max(0, event.eventIndex - eventOffsetRef.current)));
      setState(data.state || null);
      setPendingQuestion(null);
      setQueuedMessages([]);
      await runChatTurn(next, event.attachments || [], next, { state: data.state || null });
    } catch (error) {
      setHistoryStatus(`${text("编辑失败：", "Edit failed: ")}${String(error.message || error)}`);
    }
  }

  async function deletePlan() {
    if (conversationId) {
      try {
        const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/plan`, { method: "DELETE" });
        setState(data.state || null);
        return;
      } catch (error) {
        setHistoryStatus(`${text("删除失败：", "Delete failed: ")}${String(error.message || error)}`);
        return;
      }
    }
    setState(state ? { ...state, plan: null } : null);
  }

  async function respondToQuestion(response) {
    if (!pendingQuestion || busy) return;
    const payload = {
      status: response.status,
      question: pendingQuestion.question,
      selected: response.selected || [],
      text: response.text || "",
      direction: response.direction || "",
    };
    const displayMessage = formatQuestionResponseForUser(payload, text);
    setPendingQuestion(null);
    await runChatTurn(`questionResponse:\n${JSON.stringify(payload)}`, [], displayMessage);
  }

  async function respondToApproval(response) {
    if (!pendingQuestion || pendingQuestion.kind !== "approval" || busy) return;
    const payload = {
      status: response.status,
      action: pendingQuestion.action,
      text: response.text || "",
    };
    const approved = payload.status === "approved";
    const edits = pendingQuestion.source === "fileEditor" && Array.isArray(pendingQuestion.pending_edits) ? pendingQuestion.pending_edits : null;
    const shownAction = edits ? text(`应用 ${edits.length} 处文件修改：${[...new Set(edits.map((edit) => edit.path))].join(", ")}`, payload.action) : payload.action;
    const displayMessage = `${approved ? text("已批准：", "Approved: ") : text("已拒绝：", "Rejected: ")}${shownAction}${payload.text ? `\n${payload.text}` : ""}`;
    setPendingQuestion(null);
    await runChatTurn(`approvalResponse:\n${JSON.stringify(payload)}`, [], displayMessage);
  }

  async function reviewPlan(decision) {
    if (!conversationId || busy) return;
    const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/plan/decision`, {
      method: "POST",
      body: { decision },
    });
    setState(data.state || null);
    applyConversation(data, { toBottom: false });
    if (decision === "approved") {
      const nextOptions = { ...options, conversation_mode: "agent" };
      setOptions(nextOptions);
      await runChatTurn(
        "Implement the approved session plan. Verify repository facts as you work, test the result, and report any necessary deviation from the plan.",
        [],
        text("已批准计划，开始执行。", "Plan approved. Start implementation."),
        { state: data.state || null, options: nextOptions },
      );
    }
  }

  function revisePlan() {
    setOptions((current) => ({ ...current, conversation_mode: "plan" }));
    setMessage(text("请根据以下反馈修改计划：", "Revise the plan using this feedback: "));
    requestAnimationFrame(() => composerRef.current?.focus());
  }

  async function savePlanCopy() {
    if (!conversationId || busy) return;
    const result = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/plan/save-copy`, { method: "POST" });
    setHistoryStatus(`${text("计划副本已保存：", "Plan copy saved: ")}${result.path}`);
  }

  async function uploadOneFile(file) {
    const response = await fetch(`${API_BASE}/api/uploads?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: file.type ? { "Content-Type": file.type } : undefined,
      body: await file.arrayBuffer(),
    });
    if (!response.ok) {
      throw new Error(`${file.name}: ${await response.text()}`);
    }
    const data = await response.json();
    return {
      path: data.path,
      url: data.url || data.absolute_url || "",
      absolute_url: data.absolute_url || "",
      filename: data.filename || file.name,
      content_type: data.content_type || file.type || "application/octet-stream",
      size: data.size || file.size || 0,
    };
  }

  const filteredMentions = mentionOptions
    .filter((item) => {
      const text = `${item.label} ${item.token}`.toLowerCase();
      return !mentionQuery || text.includes(mentionQuery);
    })
    .slice(0, 10);

  const modelNow = conversationModel || options.model || "";
  const modelEntry = models.find((model) => model.value === modelNow);
  const modelName = modelEntry?.label || modelNow.split(":").pop() || text("默认模型", "Default model");
  const effort = options.reasoning_effort || "default";
  const agentSummary = summarizeAgents(subagents);
  const securityLevel = security?.level || "L3";
  const modeNow = normalizeConversationMode(options.conversation_mode);
  const modeInfo = {
    ask: { icon: <MessageSquare size={14} />, label: text("问答", "Ask"), hint: text("只读理解与查询", "Read-only answers") },
    plan: { icon: <ListChecks size={14} />, label: text("计划", "Plan"), hint: text("研究并等待批准", "Research, then wait for approval") },
    agent: { icon: <Bot size={14} />, label: "Agent", hint: text("执行、测试、迭代", "Implement, test, iterate") },
  };
  const updateOption = (key, value) => setOptions((current) => ({ ...current, [key]: value }));
  const nativeReady = nativeWebProviders.includes(String(modelNow).split(":")[0]);
  const currentTitle = conversations.find((item) => item.id === conversationId)?.title;

  return (
    <section className="chatLayout" style={{ gridTemplateColumns: splitColumns(historyPane, "r", null) }}>
      <aside className="historyPane" aria-hidden={historyPane.collapsed || undefined}>
        <div className="paneHeader">
          <h2>{label("history")}</h2>
          <div className="historyHeaderActions">
            <button
              className={selectMode ? "iconButton active" : "iconButton"}
              onClick={toggleSelectMode}
              type="button"
              title={selectMode ? text("退出选择", "Exit selection") : text("批量选择", "Select multiple")}
              aria-pressed={selectMode}
            >
              <ListChecks size={16} />
            </button>
            <button className="iconButton" onClick={newConversation} type="button" title={label("newChat")}>
              <Plus size={16} />
            </button>
          </div>
        </div>
        {selectMode && (
          <div className="historyBulkBar">
            <label className="historyCheck">
              <input type="checkbox" checked={allSelected} onChange={toggleSelectAll} disabled={conversations.length === 0} />
              <span>{text("全选", "Select all")}</span>
            </label>
            <small>{selectedIds.length}/{conversations.length}</small>
            {confirmBulk ? (
              <span className="bulkConfirm">
                <span>{text(`删除 ${selectedIds.length} 条？`, `Delete ${selectedIds.length}?`)}</span>
                <button className="bulkButton danger" type="button" onClick={deleteSelectedConversations}>{text("确定", "Yes")}</button>
                <button className="bulkButton" type="button" onClick={() => setConfirmBulk(false)}>{text("取消", "No")}</button>
              </span>
            ) : (
              <button className="bulkButton danger" type="button" onClick={() => setConfirmBulk(true)} disabled={selectedIds.length === 0}>
                <Trash2 size={13} />
                {text("删除", "Delete")}
              </button>
            )}
          </div>
        )}
        <div className="historyList">
          {conversations.map((item) => (
            <div
              className={`historyRow${conversationId === item.id ? " active" : ""}${selectMode ? " selecting" : ""}${selectedIds.includes(item.id) ? " checked" : ""}`}
              key={item.id}
            >
              {selectMode && (
                <input
                  className="historyCheckbox"
                  type="checkbox"
                  checked={selectedIds.includes(item.id)}
                  onChange={() => toggleSelected(item.id)}
                  aria-label={item.title || "未命名"}
                />
              )}
              {renamingId === item.id ? (
                <form className="historyRename" onSubmit={(event) => { event.preventDefault(); commitRename(); }}>
                  <input
                    autoFocus
                    value={renameDraft}
                    onChange={(event) => setRenameDraft(event.target.value)}
                    onKeyDown={(event) => { if (event.key === "Escape") setRenamingId(""); }}
                    onFocus={(event) => event.target.select()}
                    maxLength={80}
                    aria-label={text("对话名称", "Conversation name")}
                  />
                  <button className="miniButton" type="submit" title={text("保存", "Save")} aria-label={text("保存", "Save")}><Check size={14} /></button>
                  <button className="miniButton" type="button" onClick={() => setRenamingId("")} title={text("取消", "Cancel")} aria-label={text("取消", "Cancel")}><X size={14} /></button>
                </form>
              ) : (
                <button
                  className="historyItem"
                  onClick={() => (selectMode ? toggleSelected(item.id) : openConversation(item.id))}
                  type="button"
                  title={item.title || "未命名"}
                >
                  <span>{item.title || "未命名"}</span>
                </button>
              )}
              {!selectMode && renamingId !== item.id && (confirmDeleteId === item.id ? (
                <div className="historyRowActions confirming">
                  <span>{text("删除？", "Delete?")}</span>
                  <button className="miniButton dangerMini" onClick={() => deleteConversation(item.id)} type="button" title={text("确认删除", "Confirm delete")} aria-label={text("确认删除", "Confirm delete")}><Check size={14} /></button>
                  <button className="miniButton" onClick={() => setConfirmDeleteId("")} type="button" title={text("取消", "Cancel")} aria-label={text("取消", "Cancel")}><X size={14} /></button>
                </div>
              ) : (
                <div className="historyRowActions">
                  <button className="miniButton" onClick={() => startRename(item)} type="button" title={text("重命名", "Rename")} aria-label={text("重命名", "Rename")}>
                    <Pencil size={14} />
                  </button>
                  <button className="miniButton dangerMini" onClick={() => setConfirmDeleteId(item.id)} type="button" title={text("删除", "Delete")} aria-label={text("删除", "Delete")}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          ))}
        </div>
        {historyStatus && <p className="miniStatus">{historyStatus}</p>}
      </aside>
      <Resizer pane={historyPane} label={text("调整历史栏宽度", "Resize history")} />
      <div className="chatPane">
        <div className="chatHeader">
          <h2>{conversationId ? (currentTitle || text("对话", "Chat")) : label("newChat")}</h2>
          <span>{modelName}</span>
        </div>
        <div className="stream" ref={outputRef} onScroll={trackScroll}>
          {events.length === 0 && <div className="emptyState">输入消息，或用 @ 指定工具、历史、技能、记忆和知识。</div>}
          {hasMoreEvents && (
            <div className="olderLoader">
              {loadingOlder
                ? <span><span className="spinner" /> {text("正在加载更早的消息…", "Loading earlier messages...")}</span>
                : <button className="quietButton" type="button" onClick={loadOlderEvents}>{text("加载更早的消息", "Load earlier messages")}</button>}
            </div>
          )}
          <StreamEvents
            events={events}
            indexOffset={eventOffset}
            busy={busy}
            canAct={!busy && indexTrusted}
            onPreviewImage={setPreviewImage}
            onBranch={branchConversation}
            onEdit={editUserMessage}
            text={text}
          />
          <PlanReviewCard plan={busy ? null : displayPlan} busy={busy} onApprove={() => reviewPlan("approved")} onReject={() => reviewPlan("rejected")} onRevise={revisePlan} onSave={savePlanCopy} onDelete={deletePlan} text={text} />
        </div>
        <QuestionDialog
          key={`${pendingQuestion?.run_id || "saved"}:${pendingQuestion?.question || ""}`}
          question={pendingQuestion?.kind === "approval" ? null : pendingQuestion}
          busy={busy}
          onRespond={respondToQuestion}
          text={text}
        />
        <ApprovalDialog
          key={`${pendingQuestion?.run_id || "saved"}:${pendingQuestion?.action || ""}`}
          approval={pendingQuestion?.kind === "approval" ? pendingQuestion : null}
          busy={busy}
          onRespond={respondToApproval}
          text={text}
        />
        <form className="composer" onSubmit={sendMessage}>
          {queuedMessages.length > 0 && <div className="queuedMessages">{queuedMessages.map((item, index) => <div key={item.id}><span>{index + 1}. {item.displayMessage}</span><button type="button" onClick={() => setQueuedMessages((items) => items.filter((entry) => entry.id !== item.id))}><X size={14} /></button></div>)}</div>}
          {mentionOpen && filteredMentions.length > 0 && (
            <div className="mentionMenu">
              {filteredMentions.map((item) => (
                <button key={item.token} type="button" onClick={() => insertMention(item)}>
                  <span>{item.label}</span>
                  <code>{item.token}</code>
                </button>
              ))}
            </div>
          )}
          <div className="composerSurface">
            {attachments.length > 0 && (
              <div className="attachmentTray">
                {attachments.map((item) => {
                  const isImage = item.content_type?.startsWith("image/");
                  const src = item.url || item.path;
                  const href = normalizeImageSrc(src);
                  return (
                    <div className="attachmentChip" key={item.path}>
                      {isImage ? (
                        <button className="attachmentThumbButton" type="button" onClick={() => setPreviewImage({ src, alt: item.filename || "upload" })} title="View image">
                          <img className="attachmentThumb" src={href} alt={item.filename || "upload"} />
                        </button>
                      ) : (
                        <Paperclip size={14} />
                      )}
                      <span>{item.filename}</span>
                      {isImage && <a className="attachmentAction" href={href} download={imageDownloadName(item.filename || src)} title="Download image"><Download size={13} /></a>}
                      <button className="attachmentAction" type="button" onClick={() => setAttachments((current) => current.filter((candidate) => candidate.path !== item.path))} title="Remove attachment"><X size={13} /></button>
                    </div>
                  );
                })}
              </div>
            )}
            <textarea
              ref={composerRef}
              value={message}
              onChange={(event) => updateMessage(event.target.value, event.target.selectionStart)}
              onKeyDown={handleComposerKeyDown}
              onFocus={loadMentionOptions}
              placeholder="输入消息，Enter 发送，Shift+Enter 换行..."
            />
            <div className="composerBar">
              <div className="composerGroup">
                <PopPill iconOnly icon={<Plus size={16} />} title={text("添加", "Add")} width={230}>
                  {({ close }) => (
                    <div className="menuList">
                      <MenuItem icon={<Paperclip size={15} />} label={label("uploadFile")} onClick={() => { close(); fileInputRef.current?.click(); }} />
                      <MenuItem icon={<Image size={15} />} label={label("uploadImage")} onClick={() => { close(); imageInputRef.current?.click(); }} />
                      <MenuItem icon={<AtSign size={15} />} label={text("引用工具 / 历史 / 技能…", "Mention tool / history / skill…")} onClick={() => { close(); updateMessage(`${message}@`, message.length + 1); composerRef.current?.focus(); }} />
                      <MenuItem icon={<Archive size={15} />} label={label("compress")} hint={text("压缩当前上下文，完整历史保留", "Shrink the context; the full history stays")} disabled={!conversationId || busy} onClick={() => { close(); compressConversation(); }} />
                      <MenuItem icon={<RefreshCw size={15} />} label={text("重置对话", "Reset conversation")} disabled={busy} onClick={() => { close(); newConversation(); }} />
                    </div>
                  )}
                </PopPill>
                <PopPill icon={modeInfo[modeNow].icon} label={modeInfo[modeNow].label} title={text("对话模式", "Conversation mode")} width={250} caret>
                  {({ close }) => (
                    <div className="menuList">
                      {["ask", "plan", "agent"].map((mode) => (
                        <MenuItem key={mode} icon={modeInfo[mode].icon} label={modeInfo[mode].label} hint={modeInfo[mode].hint} selected={mode === modeNow} onClick={() => { updateOption("conversation_mode", mode); close(); }} />
                      ))}
                    </div>
                  )}
                </PopPill>
                <PopPill
                  icon={<Globe size={14} />}
                  label={webModeLabel(normalizeWebMode(options.web_search_mode), text)}
                  title={text("联网", "Web access")}
                  tone={normalizeWebMode(options.web_search_mode) === "off" ? "" : "on"}
                  width={330}
                >
                  <WebAccessControl options={options} update={updateOption} nativeReady={nativeReady} text={text} />
                </PopPill>
                <PopPill
                  icon={<ShieldCheck size={14} />}
                  label={securityLevel}
                  title={text("安全级别", "Security level")}
                  tone={{ L1: "safe", L2: "sandbox", L3: "full" }[securityLevel]}
                  width={340}
                >
                  <SecurityControl security={security} error={securityError} onChange={changeSecurity} text={text} />
                </PopPill>
                <PopPill iconOnly icon={<SlidersHorizontal size={15} />} title={text("更多设置", "More settings")} width={460}>
                  <MoreSettings options={options} update={updateOption} text={text} clearState={newConversation} />
                </PopPill>
              </div>
              <div className="composerGroup right">
                {busy && <span className="busyTag" title={text("本轮已运行", "Running for")}><span className="spinner" />{formatElapsed(elapsed)}</span>}
                <PopPill
                  icon={agentSummary.running > 0 ? <span className="runningDot" /> : <span className={`statusDot tone-${agentSummary.tone || "idle"}`} />}
                  label={text(`${agentSummary.total} 个代理`, `${agentSummary.total} agents`)}
                  title={text("子代理", "Sub-agents")}
                  tone={agentSummary.tone}
                  align="right"
                  width={440}
                >
                  <div className="popBody">
                    <SubagentPanel agents={subagents} tasks={agentTasks} conversationId={conversationId} onCancel={cancelSubagent} text={text} />
                    <div className="popSection">
                    <div className="formGrid">
                      <label className="field">
                        <span>{text("子代理模型", "Sub-agent model")}</span>
                        <select value={options.subagent_model || ""} onChange={(event) => updateOption("subagent_model", event.target.value)}>
                          <option value="">{text("跟随主模型", "Same as main")}</option>
                          {models.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
                        </select>
                      </label>
                    </div>
                    <StepSlider
                      title={text("子代理思考强度", "Sub-agent thinking")}
                      steps={reasoningSteps(text)}
                      value={options.subagent_reasoning_effort || "default"}
                      onChange={(value) => updateOption("subagent_reasoning_effort", value)}
                    />
                    <p className="popFoot">{text("派发任务时，Agent 也可以按难度自己为每个任务选模型。", "When spawning, the agent can also pick a model per task by difficulty.")}</p>
                    </div>
                  </div>
                </PopPill>
                <PopPill
                  icon={<Brain size={14} />}
                  label={<><span className="pillModel">{modelName}</span>{effort !== "default" && <span className="pillTag">{reasoningLabel(effort, text)}</span>}</>}
                  title={text("模型与思考强度", "Model and thinking")}
                  align="right"
                  width={340}
                  className="modelPill"
                >
                  {({ close }) => (
                    <div className="popBody">
                      <div className="menuList scroll">
                        <MenuItem label={text("默认模型", "Default model")} selected={!modelNow} onClick={() => { setConversationModel(""); close(); }} />
                        {models.map((model) => (
                          <MenuItem key={model.value} label={model.label} selected={model.value === modelNow} onClick={() => { setConversationModel(model.value); close(); }} />
                        ))}
                      </div>
                      <StepSlider
                        title={text("思考强度", "Thinking effort")}
                        steps={reasoningSteps(text)}
                        value={effort}
                        onChange={(value) => updateOption("reasoning_effort", value)}
                      />
                    </div>
                  )}
                </PopPill>
                <PopPill
                  icon={<ShieldCheck size={13} />}
                  label={(approvalModes(text).find((mode) => mode.value === options.file_editor_approval) || approvalModes(text)[0]).short}
                  title={text("批准方式", "Approval mode")}
                  active={options.file_editor_approval === "auto"}
                  align="right"
                  width={320}
                >
                  {({ close }) => (
                    <div className="menuList">
                      {approvalModes(text).map((mode) => (
                        <MenuItem key={mode.value} label={mode.label} hint={mode.hint} selected={options.file_editor_approval === mode.value} onClick={() => { updateOption("file_editor_approval", mode.value); close(); }} />
                      ))}
                    </div>
                  )}
                </PopPill>
                {busy && (
                  <Pill
                    label={inputMode === "insert" ? text("打断", "Interrupt") : text("排队", "Queue")}
                    title={text("运行中发送新消息的方式：点击切换", "What sending does while running: click to switch")}
                    onClick={() => setInputMode((mode) => (mode === "insert" ? "queue" : "insert"))}
                  />
                )}
                {busy && <button className="roundButton" type="button" onClick={togglePause} title={paused ? text("继续", "Resume") : text("暂停", "Pause")}>{paused ? <Play size={15} /> : <Pause size={15} />}</button>}
                {busy && <button className="roundButton danger" type="button" onClick={stopOutput} title={label("stop")}><Square size={13} /></button>}
                <button className="sendButton" type="submit" disabled={!message.trim() && attachments.length === 0} title={busy ? (inputMode === "insert" ? text("打断并发送", "Interrupt and send") : text("加入队列", "Add to queue")) : label("send")} aria-label={label("send")}><ArrowUp size={17} /></button>
              </div>
            </div>
          </div>
          <input ref={fileInputRef} type="file" className="hiddenInput" onChange={uploadSelectedFile} multiple />
          <input ref={imageInputRef} type="file" accept="image/*" className="hiddenInput" onChange={uploadSelectedFile} multiple />
        </form>
        <ImagePreview image={previewImage} onClose={() => setPreviewImage(null)} />
      </div>
    </section>
  );
}

const WEB_MODES = ["off", "tool", "native", "both"];

// "auto" is the older name for "tool".
function normalizeWebMode(value) {
  if (value === "auto") return "tool";
  return WEB_MODES.includes(value) ? value : "tool";
}

function webModeLabel(mode, text) {
  return { off: text("关闭", "Off"), tool: text("工具", "Tool"), native: text("原生", "Native"), both: text("两者", "Both") }[mode];
}

function reasoningSteps(text) {
  return [
    { value: "default", label: text("默认", "Default"), hint: text("不向供应商发送思考参数，使用模型自己的默认值。", "Send no reasoning parameter; the model keeps its own default.") },
    { value: "off", label: text("关闭", "Off"), hint: text("尽量不思考（供应商支持时）。", "Ask for no thinking where the provider supports it.") },
    { value: "low", label: text("低", "Low"), hint: text("少量思考，更快更省。", "A little thinking: faster and cheaper.") },
    { value: "medium", label: text("中", "Medium"), hint: text("均衡。", "Balanced.") },
    { value: "high", label: text("高", "High"), hint: text("深入思考，适合复杂任务。", "Deep thinking for hard tasks.") },
    { value: "max", label: text("最大", "Max"), hint: text("最大思考强度，最慢也最贵。", "Maximum effort: slowest and most expensive.") },
  ];
}

function reasoningLabel(value, text) {
  return reasoningSteps(text).find((step) => step.value === value)?.label || "";
}

function securitySteps(text) {
  return [
    { value: "L1", label: text("L1 只读", "L1 Read"), hint: text("只读且无害：读取 agent 自己的数据、RAG、历史、联网搜索。不执行任何代码，不能修改文件。", "Read-only and harmless: the agent's own data, RAG, history and web search. No code runs and nothing is edited.") },
    { value: "L2", label: text("L2 隔离", "L2 Sandbox"), hint: text("所有工具可用，但 python、自定义工具和 stdio MCP 以无权限的系统用户运行，读不到密钥、配置和源码。需要 Linux 且后端以 root 启动（Docker 部署）。", "Every tool, but python, custom tools and stdio MCP run as an unprivileged OS user that cannot read secrets, configuration or source. Needs Linux and a backend started as root (Docker).") },
    { value: "L3", label: text("L3 完全", "L3 Full"), hint: text("所有工具以后端自身的权限运行，等同于后端进程能做的一切。", "Everything runs with the backend's own rights: whatever the backend process can do.") },
  ];
}

function WebAccessControl({ options, update, nativeReady, text }) {
  const mode = normalizeWebMode(options.web_search_mode);
  const steps = WEB_MODES.map((value) => ({
    value,
    label: webModeLabel(value, text),
    hint: {
      off: text("不联网。", "No web access."),
      tool: text("模型调用 webSearch 工具搜索（DuckDuckGo / SearXNG / Tavily）。", "The model calls the webSearch tool (DuckDuckGo / SearXNG / Tavily)."),
      native: text("使用供应商自带的联网：OpenRouter / 百炼在请求里开启搜索；DeepSeek 通过官方 Search / Fetch 接口提供 nativeWebSearch 与 webFetch 工具。不提供 webSearch 工具。", "Use the provider's own web access: OpenRouter / DashScope switch search on in the request; DeepSeek offers nativeWebSearch and webFetch tools through its Search / Fetch API. No webSearch tool."),
      both: text("同时提供 webSearch 工具和供应商原生联网。", "Offer the webSearch tool and the provider's native web access."),
    }[value],
  }));
  return (
    <div className="popBody">
      <StepSlider title={text("联网", "Web access")} steps={steps} value={mode} onChange={(value) => update("web_search_mode", value)} />
      {(mode === "native" || mode === "both") && !nativeReady && (
        <p className="notice warn">
          {text("当前模型的供应商还没有配置原生联网搜索：请到「模型」页，在该供应商下选择“原生联网搜索”。", "This model's provider has no native web search configured: choose one under that provider on the Models page.")}
        </p>
      )}
      <div className="formGrid">
        <SelectField label={text("搜索引擎", "Search engine")} value={options.web_search_provider} onChange={(value) => update("web_search_provider", value)} values={choices("searchEngine", text)} />
        <SelectField label={text("失败自动切换", "Fallback")} value={String(options.web_search_auto_switch)} onChange={(value) => update("web_search_auto_switch", value === "true")} values={choices("onOff", text)} />
      </div>
    </div>
  );
}

function SecurityControl({ security, error, onChange, text }) {
  const steps = securitySteps(text);
  const level = security?.level || "L3";
  const l2 = security?.l2;
  return (
    <div className="popBody">
      <StepSlider title={text("安全级别", "Security level")} steps={steps} value={level} onChange={onChange} tone={level.toLowerCase()} />
      {l2 && !l2.available && <p className="notice warn">{text("L2 当前不可用：", "L2 is unavailable here: ")}{l2.reason}</p>}
      {error && error !== l2?.reason && <p className="notice error">{error}</p>}
      <p className="popFoot">{text("随时可调，立即生效（包括正在运行的回合）。只能在这里修改，模型和后端代码都改不了。", "Changes apply at once, mid-turn included. Only this slider can change it - not the model, not agent code.")}</p>
    </div>
  );
}

function MoreSettings({ options, update, text, clearState }) {
  return (
    <div className="popBody">
      <StepSlider
        title={text("提问强度", "Clarification")}
        steps={[
          { value: "off", label: text("跳过", "Skip") },
          { value: "light", label: text("轻度", "Light") },
          { value: "heavy", label: text("重度", "Heavy") },
        ]}
        value={options.question_mode}
        onChange={(value) => update("question_mode", value)}
      />
      <div className="formGrid three">
        <SelectField label="RAG" value={options.rag_mode} onChange={(value) => update("rag_mode", value)} values={choices("rag", text)} />
        <SelectField label="HTTP" value={options.curl_mode} onChange={(value) => update("curl_mode", value)} values={choices("switch", text)} />
        <SelectField label="Python" value={options.python_mode} onChange={(value) => update("python_mode", value)} values={choices("switch", text)} />
        <SelectField label={text("文件阅读", "File reader")} value={options.file_reader_mode} onChange={(value) => update("file_reader_mode", value)} values={choices("switch", text)} />
        <SelectField label={text("文件编辑", "File editor")} value={options.file_editor_mode} onChange={(value) => update("file_editor_mode", value)} values={choices("switch", text)} />
        <SelectField label={text("编辑批准", "Edit approval")} value={options.file_editor_approval} onChange={(value) => update("file_editor_approval", value)} values={choices("approval", text)} />
        <SelectField label="MCP" value={options.mcp_mode} onChange={(value) => update("mcp_mode", value)} values={choices("switch", text)} />
        <SelectField label={text("历史", "History")} value={options.history_mode} onChange={(value) => update("history_mode", value)} values={choices("switch", text)} />
        <SelectField label={text("自动化", "Automation")} value={options.automation_mode} onChange={(value) => update("automation_mode", value)} values={choices("switch", text)} />
        <label className="field">
          <span>{text("工具轮次", "Tool rounds")}</span>
          <input type="number" min="-1" value={options.max_tool_rounds} onChange={(event) => update("max_tool_rounds", Number(event.target.value))} />
        </label>
      </div>
      <small className="muted">{text("工具轮次：-1 = 无限制，0 = 禁用工具", "Tool rounds: -1 = unlimited, 0 = tools disabled")}</small>
      <div className="checkRow">
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_memory} onChange={(event) => update("rag_include_memory", event.target.checked)} /><span>{text("记忆", "Memory")}</span></label>
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_skills} onChange={(event) => update("rag_include_skills", event.target.checked)} /><span>{text("技能", "Skills")}</span></label>
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_knowledge} onChange={(event) => update("rag_include_knowledge", event.target.checked)} /><span>{text("知识", "Knowledge")}</span></label>
        <label className="checkLine"><input type="checkbox" checked={Boolean(options.developer_mode)} onChange={(event) => update("developer_mode", event.target.checked)} /><span>{text("开发者模式", "Developer mode")}</span></label>
      </div>
      <label className="field">
        <span>{text("额外提示", "Extra prompt")}</span>
        <textarea className="smallTextArea" value={options.system_prompt || ""} onChange={(event) => update("system_prompt", event.target.value)} placeholder={text("仅当前对话追加到首个 system prompt", "Append only to the first system prompt in this chat")} />
      </label>
      <button className="quietButton" type="button" onClick={clearState}>
        <RefreshCw size={14} />
        <span>{text("重置对话", "Reset conversation")}</span>
      </button>
    </div>
  );
}

// What the composer's approval pill offers. No mode involves a model except AI review.
function approvalModes(text) {
  return [
    {
      value: "auto",
      short: text("自动批准", "Auto"),
      label: text("自动执行", "Run automatically"),
      hint: text("文件修改、Python、MCP 等直接执行，不经过任何审查。", "File edits, Python, MCP and the rest run directly, without any review."),
    },
    {
      value: "manual",
      short: text("手动批准", "Manual"),
      label: text("手动批准文件修改", "Approve file edits"),
      hint: text("每次改文件先暂停并显示 diff，你批准后由系统按原样写入；其他操作照常执行。", "Each file edit pauses with its diff and the system writes exactly that when you approve; other calls run as usual."),
    },
    {
      value: "aiReview",
      short: text("AI 审查", "AI review"),
      label: text("AI 审查", "AI review"),
      hint: text("高风险操作（Python、写文件、MCP 调用、改设置等）先交给模型审查，拒绝则不执行。", "High-risk calls (Python, file writes, MCP calls, settings…) are checked by a model first and skipped when it refuses."),
    },
    {
      value: "readOnly",
      short: text("只读", "Read-only"),
      label: text("禁止写文件", "No file writes"),
      hint: text("文件编辑只能读，不能写；其他操作照常执行。", "The file editor can only read; other calls run as usual."),
    },
  ];
}

// Option labels for every fixed-choice select, in both UI languages: [value, 中文, English].
const CHOICES = {
  switch: [["off", "关闭", "Off"], ["auto", "自动", "Auto"]],
  rag: [["off", "关闭", "Off"], ["on", "开启", "On"], ["auto", "自动", "Auto"]],
  onOff: [["true", "开启", "On"], ["false", "关闭", "Off"]],
  searchEngine: [["duckduckgo", "DuckDuckGo", "DuckDuckGo"], ["searxng", "SearXNG", "SearXNG"], ["tavily", "Tavily", "Tavily"]],
  approval: [["manual", "手动批准", "Manual"], ["auto", "自动", "Auto"], ["aiReview", "AI 审查", "AI review"], ["readOnly", "只读", "Read-only"]],
  automation: [["reminder", "提醒", "Reminder"], ["llm", "模型任务", "LLM task"], ["script", "Python 脚本", "Python script"], ["mcp", "MCP 调用", "MCP call"], ["configureMcp", "配置 MCP", "Configure MCP"]],
  transport: [["streamable_http", "Streamable HTTP（远程）", "Streamable HTTP (remote)"], ["stdio", "stdio（本地命令）", "stdio (local command)"]],
};

function choices(kind, text) {
  return CHOICES[kind].map(([value, zh, en]) => [value, text(zh, en)]);
}

// values: plain strings or [value, label] pairs. A bare value is shown with a capital letter, never raw.
function SelectField({ label, value, onChange, values, icon = null }) {
  return (
    <label className="field">
      <span>{icon}{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {values.map((item) => {
          const [optionValue, optionLabel] = Array.isArray(item) ? item : [item, item.charAt(0).toUpperCase() + item.slice(1)];
          return <option key={optionValue} value={optionValue}>{optionLabel}</option>;
        })}
      </select>
    </label>
  );
}

function formatElapsed(seconds) {
  return seconds >= 60 ? `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}` : `${seconds}s`;
}

function useElapsed(active) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active) {
      setSeconds(0);
      return undefined;
    }
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [active]);
  return seconds;
}

function StreamEvents({ events, indexOffset = 0, busy = false, canAct = false, onPreviewImage, onBranch, onEdit, text }) {
  const turns = [];
  let current = [];
  for (const [localIndex, rawEvent] of events.entries()) {
    // eventIndex is the position in the saved conversation, not in the loaded window.
    const event = { ...rawEvent, eventIndex: indexOffset + localIndex };
    if (event.type === "user" && current.length > 0) {
      turns.push(current);
      current = [];
    }
    current.push(event);
  }
  if (current.length > 0) turns.push(current);
  return turns.map((turn, index) => (
    <StreamTurn
      key={index}
      events={turn}
      running={busy && index === turns.length - 1}
      onPreviewImage={onPreviewImage}
      onBranch={onBranch}
      onEdit={onEdit}
      canAct={canAct}
      text={text}
    />
  ));
}

function PlanReviewCard({ plan, busy, onApprove, onReject, onRevise, onSave, onDelete, text }) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  if (!plan?.content || plan.status === "rejected") return null;
  const ready = plan.status === "ready";
  return (
    <section className={`planReview plan-${plan.status || "draft"}`}>
      <header>
        <div>
          <span>{text("会话计划", "SESSION PLAN")}</span>
          <h3>{plan.name || text("实施计划", "Implementation plan")}</h3>
        </div>
        <strong>{String(plan.status || "draft").toUpperCase()}</strong>
      </header>
      <MarkdownText text={plan.content} />
      <div className="planActions">
        <button className="secondaryButton" type="button" onClick={onRevise} disabled={busy}>{text("继续修改", "Revise")}</button>
        <button className="secondaryButton" type="button" onClick={onSave} disabled={busy}><Save size={15} /><span>{text("保存副本", "Save copy")}</span></button>
        {confirmingDelete ? (
          <>
            <button className="secondaryButton" type="button" onClick={() => setConfirmingDelete(false)}>{text("取消", "Cancel")}</button>
            <button className="secondaryButton dangerButton" type="button" onClick={() => { setConfirmingDelete(false); onDelete(); }} disabled={busy}><Trash2 size={15} /><span>{text("确认删除", "Confirm delete")}</span></button>
          </>
        ) : (
          <button className="secondaryButton dangerButton" type="button" onClick={() => setConfirmingDelete(true)} disabled={busy}><Trash2 size={15} /><span>{text("删除计划", "Delete plan")}</span></button>
        )}
        {ready && <button className="secondaryButton dangerButton" type="button" onClick={onReject} disabled={busy}>{text("拒绝", "Reject")}</button>}
        {ready && <button className="primaryButton" type="button" onClick={onApprove} disabled={busy}><Check size={16} /><span>{text("批准并执行", "Approve and implement")}</span></button>}
      </div>
    </section>
  );
}

function QuestionDialog({ question, busy, onRespond, text }) {
  const [selected, setSelected] = useState([]);
  const [answer, setAnswer] = useState("");
  const [redirecting, setRedirecting] = useState(false);
  const [direction, setDirection] = useState("");
  const [minimized, setMinimized] = useState(false);
  if (!question) return null;

  if (minimized) {
    return (
      <button className="questionMinimized" type="button" onClick={() => setMinimized(false)}>
        <span><strong>{text("等待确认", "Input needed")}</strong>{question.question}</span>
        <Maximize2 size={17} />
      </button>
    );
  }

  const options = Array.isArray(question.options) ? question.options : [];
  const canAnswer = selected.length > 0 || answer.trim().length > 0;

  function toggleOption(option) {
    if (question.multiple) {
      setSelected((items) => items.includes(option) ? items.filter((item) => item !== option) : [...items, option]);
      return;
    }
    setSelected([option]);
  }

  function submitAnswer(event) {
    event.preventDefault();
    if (!canAnswer || busy) return;
    onRespond({ status: "answered", selected, text: answer.trim() });
  }

  function submitDirection(event) {
    event.preventDefault();
    if (!direction.trim() || busy) return;
    onRespond({ status: "redirected", direction: direction.trim() });
  }

  return (
    <div className="questionBackdrop">
      <section className="questionDialog" role="dialog" aria-modal="true" aria-labelledby="question-title">
        <header className="questionHeader">
          <div>
            <span>{text("需要你的确认", "YOUR INPUT")}</span>
            <h2 id="question-title">{redirecting ? text("改变方向", "Change direction") : (question.title || text("确认下一步", "Confirm next step"))}</h2>
          </div>
          <div className="questionHeaderActions">
            <button className="iconButton neutral" type="button" onClick={() => setMinimized(true)} title={text("缩小", "Minimize")}>
              <Minus size={17} />
            </button>
            <button
              className="iconButton neutral"
              type="button"
              onClick={() => onRespond({ status: "declined" })}
              disabled={busy}
              title={text("拒绝回答", "Decline")}
            >
              <X size={17} />
            </button>
          </div>
        </header>

        {redirecting ? (
          <form onSubmit={submitDirection}>
            <p className="questionText">{text("这个问题不适用，或你希望任务改走另一条路线。", "Explain the new direction or why this question does not apply.")}</p>
            <textarea
              autoFocus
              value={direction}
              onChange={(event) => setDirection(event.target.value)}
              placeholder={text("告诉 AI 接下来应该怎么做…", "Tell the AI what to do instead...")}
            />
            <div className="questionActions">
              <button className="secondaryButton" type="button" onClick={() => setRedirecting(false)} disabled={busy}>{text("返回问题", "Back")}</button>
              <button className="secondaryButton" type="button" onClick={() => onRespond({ status: "declined" })} disabled={busy}>{text("拒绝回答", "Decline")}</button>
              <button className="primaryButton" type="submit" disabled={!direction.trim() || busy}>{text("提交新方向", "Submit direction")}</button>
            </div>
          </form>
        ) : (
          <form onSubmit={submitAnswer}>
            <p className="questionText">{question.question}</p>
            {options.length > 0 && (
              <div className="questionOptions">
                {options.map((option) => (
                  <label className={selected.includes(option) ? "questionOption selected" : "questionOption"} key={option}>
                    <input
                      type={question.multiple ? "checkbox" : "radio"}
                      name="agent-question"
                      checked={selected.includes(option)}
                      onChange={() => toggleOption(option)}
                    />
                    <span>{option}</span>
                  </label>
                ))}
              </div>
            )}
            <label className="questionAnswer">
              <span>{options.length > 0 ? text("补充说明或填写其他答案", "Add details or another answer") : text("你的回答", "Your answer")}</span>
              <textarea
                autoFocus={options.length === 0}
                value={answer}
                onChange={(event) => setAnswer(event.target.value)}
                placeholder={question.placeholder || text("输入内容…", "Type your answer...")}
              />
            </label>
            <div className="questionActions">
              <button className="secondaryButton" type="button" onClick={() => onRespond({ status: "declined" })} disabled={busy}>{text("拒绝回答", "Decline")}</button>
              <button className="secondaryButton" type="button" onClick={() => setRedirecting(true)} disabled={busy}>{text("改变方向", "Change direction")}</button>
              <button className="primaryButton" type="submit" disabled={!canAnswer || busy}>{text("提交回答", "Submit answer")}</button>
            </div>
          </form>
        )}
      </section>
    </div>
  );
}

const AGENT_ACTIVE = new Set(["running", "queued"]);

function agentStatusLabel(text) {
  return {
    queued: text("排队中", "Queued"), running: text("运行中", "Running"), done: text("已完成", "Done"),
    failed: text("失败", "Failed"), cancelled: text("已取消", "Cancelled"), timeout: text("超时", "Timed out"),
    interrupted: text("已中断", "Interrupted"),
  };
}

// What the "N agents" pill shows: how many, and the colour that says how they are doing.
function summarizeAgents(agents) {
  const list = Object.values(agents);
  const running = list.filter((agent) => AGENT_ACTIVE.has(agent.status)).length;
  const bad = list.filter((agent) => ["failed", "timeout", "interrupted"].includes(agent.status)).length;
  let tone = "";
  if (running > 0) tone = "running";
  else if (bad > 0) tone = "failed";
  else if (list.length > 0) tone = "done";
  return { total: list.length, running, tone };
}

function SubagentPanel({ agents, tasks = [], conversationId, onCancel, text }) {
  const list = Object.values(agents).sort((a, b) => (a.startedAt || 0) - (b.startedAt || 0));
  const [expanded, setExpanded] = useState("");
  const [detail, setDetail] = useState(null);
  const [now, setNow] = useState(Date.now());
  const runningCount = list.filter((agent) => AGENT_ACTIVE.has(agent.status)).length;

  useEffect(() => {
    if (runningCount === 0) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [runningCount]);

  useEffect(() => {
    if (!expanded) {
      setDetail(null);
      return undefined;
    }
    let stale = false;
    fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/subagents/${encodeURIComponent(expanded)}`)
      .then((data) => { if (!stale) setDetail(data); })
      .catch(() => {});
    return () => { stale = true; };
  }, [expanded, conversationId, list.find((agent) => agent.id === expanded)?.status]);

  if (list.length === 0 && tasks.length === 0) {
    return <p className="popEmpty">{text("还没有后台任务。Agent 派发子代理后会显示在这里。", "No background agents yet. They appear here once the agent spawns some.")}</p>;
  }
  const elapsed = (agent) => {
    const seconds = AGENT_ACTIVE.has(agent.status) ? Math.max(0, now / 1000 - (agent.startedAt || now / 1000)) : agent.elapsedSeconds || 0;
    return seconds >= 60 ? `${Math.floor(seconds / 60)}m${String(Math.floor(seconds % 60)).padStart(2, "0")}s` : `${Math.floor(seconds)}s`;
  };
  const statusLabel = agentStatusLabel(text);
  const taskLabel = { done: text("完成", "done"), failed: text("失败", "failed"), cancelled: text("已取消", "cancelled"), running: text("进行中", "running"), pending: text("待办", "pending") };

  return (
    <div className="subagentList">
      {tasks.length > 0 && (
        <ol className="taskList" aria-label={text("任务列表", "Task list")}>
          {tasks.map((task) => (
            <li className={`taskItem task-${task.status}`} key={task.id}>
              <span className="taskMark" aria-hidden="true">{{ done: "✓", failed: "✗", cancelled: "–", running: "…", pending: "○" }[task.status] || "○"}</span>
              <span className="taskTitle">{task.title}</span>
              <small>{taskLabel[task.status]}</small>
            </li>
          ))}
        </ol>
      )}
      {list.map((agent) => {
        const active = AGENT_ACTIVE.has(agent.status);
        return (
          <article className={`subagentCard status-${agent.status}`} key={agent.id}>
            <div className="subagentHead">
              <button className="subagentTitle" type="button" onClick={() => setExpanded(expanded === agent.id ? "" : agent.id)}>
                {active ? <span className="runningDot" /> : <span className="subagentStatusDot" />}
                <b>{agent.name}</b>
                {agent.model && <em className="bgBadge" title={text("这个代理使用的模型", "Model this agent uses")}>{String(agent.model).split(":").pop()}</em>}
                {agent.background && <em className="bgBadge" title={text("有意后台运行：不会随对话结束被终止", "Deliberately in the background: not killed when the turn ends")}>{text("后台", "BG")}</em>}
                <span className="statusTag">{statusLabel[agent.status] || agent.status}</span>
                <small>{elapsed(agent)}{agent.rounds ? ` · ${agent.rounds} ${text("步", "steps")}` : ""}</small>
              </button>
              {active && (
                <button className="miniButton dangerMini" type="button" onClick={() => onCancel(agent.id)} title={text("取消这个任务", "Cancel this agent")} aria-label={text("取消", "Cancel")}>
                  <X size={14} />
                </button>
              )}
            </div>
            {active && agent.activity && <p className="subagentLine">{agent.activity}</p>}
            {!active && agent.error && <p className="subagentLine error">{agent.killReason ? `${text("已终止：", "Killed: ")}${agent.killReason}` : agent.error}</p>}
            {!active && !agent.error && agent.result && expanded !== agent.id && <p className="subagentLine">{agent.result}</p>}
            {expanded === agent.id && (
              <div className="subagentDetail">
                {detail?.prompt && <details><summary>{text("任务", "Task")}</summary><MarkdownText text={detail.prompt} /></details>}
                {detail?.events?.length > 0 && (
                  <ul className="subagentSteps">
                    {detail.events.filter((item) => item.type === "tool_call").slice(-12).map((item, index) => <li key={index}>{item.text}</li>)}
                  </ul>
                )}
                {detail?.fullResult ? <MarkdownText text={detail.fullResult} /> : <p className="subagentLine">{text("尚无结果。", "No result yet.")}</p>}
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}

function ApprovalDialog({ approval, busy, onRespond, text }) {
  const [note, setNote] = useState("");
  const [minimized, setMinimized] = useState(false);
  if (!approval) return null;
  const risk = ["low", "medium", "high"].includes(approval.risk) ? approval.risk : "medium";
  const edits = approval.source === "fileEditor" && Array.isArray(approval.pending_edits) ? approval.pending_edits : null;
  const editPaths = edits ? [...new Set(edits.map((edit) => edit.path))].join(", ") : "";
  const actionText = edits ? text(`应用 ${edits.length} 处文件修改：${editPaths}`, `Apply ${edits.length} file edit${edits.length === 1 ? "" : "s"}: ${editPaths}`) : approval.action;
  const titleText = approval.title || (edits ? text("批准文件修改", "Approve file edits") : text("确认执行操作", "Confirm action"));
  const riskLabel = { low: text("低风险", "Low risk"), medium: text("中等风险", "Medium risk"), high: text("高风险", "High risk") }[risk];

  if (minimized) {
    return (
      <button className="questionMinimized" type="button" onClick={() => setMinimized(false)}>
        <span><strong>{text("等待批准", "Approval needed")}</strong>{actionText}</span>
        <Maximize2 size={17} />
      </button>
    );
  }

  return (
    <div className="questionBackdrop">
      <section className={`questionDialog approvalDialog risk-${risk}`} role="dialog" aria-labelledby="approval-title">
        <header className="questionHeader">
          <div>
            <span><ShieldCheck size={12} /> {text("需要你的批准", "APPROVAL NEEDED")} · <b className={`riskBadge risk-${risk}`}>{riskLabel}</b></span>
            <h2 id="approval-title">{titleText}</h2>
          </div>
          <div className="questionHeaderActions">
            <button className="iconButton neutral" type="button" onClick={() => setMinimized(true)} title={text("缩小", "Minimize")}>
              <Minus size={17} />
            </button>
          </div>
        </header>
        <p className="questionText">{actionText}</p>
        {approval.details && <div className="approvalDetails"><MarkdownText text={approval.details} /></div>}
        <label className="questionAnswer">
          <span>{text("补充说明（可选，会一并发给 AI）", "Note for the AI (optional)")}</span>
          <textarea
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder={text("例如：只处理 build/ 目录；或说明拒绝的原因…", "e.g. only touch build/, or why you are rejecting...")}
          />
        </label>
        <div className="questionActions">
          <button className="secondaryButton dangerButton" type="button" onClick={() => onRespond({ status: "rejected", text: note.trim() })} disabled={busy}>
            <X size={15} /><span>{text("拒绝", "Reject")}</span>
          </button>
          <button className="primaryButton" type="button" onClick={() => onRespond({ status: "approved", text: note.trim() })} disabled={busy}>
            <Check size={15} /><span>{text("批准", "Approve")}</span>
          </button>
        </div>
      </section>
    </div>
  );
}

function StreamTurn({ events, running = false, canAct = false, onPreviewImage, onBranch, onEdit, text }) {
  const [expanded, setExpanded] = useState(false);
  const assistantIndex = findLastIndex(events, (event) => event.type === "assistant");
  const stoppedIndex = findLastIndex(events, (event) => event.type === "stopped");
  const errorIndex = findLastIndex(events, (event) => event.type === "error" && event.terminal);
  const finalIndex = Math.max(assistantIndex, stoppedIndex, errorIndex);
  const completed = finalIndex >= 0;
  const userEvents = events.filter((event) => event.type === "user");
  const finalEvent = completed ? events[finalIndex] : null;
  const operationEvents = events.filter((event, index) => index !== finalIndex && event.type !== "user");
  const currentWork = currentOperation(operationEvents);
  const active = !completed && (running || operationEvents.length > 0);

  useEffect(() => {
    if (completed) setExpanded(false);
  }, [completed]);

  return (
    <div className="streamTurn">
      {userEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} onPreviewImage={onPreviewImage} onBranch={canAct ? onBranch : undefined} onEdit={canAct ? onEdit : undefined} />)}
      {(operationEvents.length > 0 || active) && (
        <section className="operationGroup">
          <details className="operationDetails" open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
            <summary>
              <ChevronRight size={14} className="operationChevron" />
              {running && !completed && <span className="runningDot" />}
              <span>{completed ? text("执行过程", "Activity") : currentWork.attention?.type === "question_required" ? text("等待回答", "Waiting for answer") : currentWork.attention?.type === "approval_request" ? text("等待批准", "Waiting for approval") : currentWork.attention?.type === "approval_required" ? text("等待批准", "Waiting for approval") : text("进行中", "Working")}</span>
              <small>{operationEvents.length} {text("条记录", "events")}</small>
            </summary>
            {operationEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} compact onPreviewImage={onPreviewImage} onBranch={canAct ? onBranch : undefined} />)}
          </details>
          {active && !expanded && (
            <div className="operationPreview">
              {currentWork.thinking ? <MarkdownText text={currentWork.thinking} onPreviewImage={onPreviewImage} /> : currentWork.tools.length === 0 && <p className="operationPending">{text("正在思考…", "Thinking...")}</p>}
              {currentWork.tools.length > 0 && (
                <ul className="operationTools" aria-label={text("当前步骤的工具调用", "Current step tool calls")}>
                  {currentWork.tools.map((event, index) => <li key={event.id || `${event.tool}-${index}`}><Wrench size={14} aria-hidden="true" /><span>{toolPreviewText(event)}</span></li>)}
                </ul>
              )}
              {currentWork.attention && <p className={`operationAttention${currentWork.attention.type === "error" ? " isError" : ""}`}>{currentWork.attention.text || currentWork.attention.question || text("等待确认", "Waiting for confirmation")}</p>}
            </div>
          )}
        </section>
      )}
      {finalEvent && String(finalEvent.text || "").trim() && <StreamEvent event={finalEvent} onPreviewImage={onPreviewImage} onBranch={canAct ? onBranch : undefined} />}
    </div>
  );
}

function StreamEvent({ event, compact = false, onPreviewImage, onBranch, onEdit }) {
  const type = event.type || "event";
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const fallback = event.text || event.query || event.url || event.question || event.action || "";
  const body = stringifyEventText(event, fallback);
  const images = extractLooseImages(body);
  const canFork = Boolean(onBranch) && Number.isInteger(event.eventIndex);
  const isWake = type === "user" && event.source === "wake";
  const canEdit = type === "user" && !isWake && Boolean(onEdit) && Number.isInteger(event.eventIndex);

  function startEdit() {
    setDraft(event.text || "");
    setEditing(true);
  }

  function submitEdit(submitEvent) {
    submitEvent.preventDefault();
    if (!draft.trim()) return;
    setEditing(false);
    onEdit(event, draft);
  }

  return (
    <article className={`event event-${type}${isWake ? " wakeNotice" : ""}${compact ? " compact" : ""}${editing ? " isEditing" : ""}`}>
      <div className="eventType">{isWake ? "BACKGROUND" : eventLabel(type)}</div>
      {(canEdit || canFork) && !editing && (
        <div className="eventActions">
          {canEdit && <button className="branchButton" type="button" onClick={startEdit} title="编辑并从这里重新生成 / Edit and regenerate from here" aria-label="Edit"><Pencil size={14} /></button>}
          {canFork && <button className="branchButton" type="button" onClick={() => onBranch(event)} title="从这里分叉为新对话 / Fork into a new conversation" aria-label="Fork"><GitBranch size={14} /></button>}
        </div>
      )}
      {editing ? (
        <form className="editMessage" onSubmit={submitEdit}>
          <textarea
            autoFocus
            value={draft}
            onChange={(changeEvent) => setDraft(changeEvent.target.value)}
            onKeyDown={(keyEvent) => {
              if (keyEvent.key === "Enter" && (keyEvent.ctrlKey || keyEvent.metaKey)) keyEvent.currentTarget.form?.requestSubmit();
              if (keyEvent.key === "Escape") setEditing(false);
            }}
          />
          <div className="editMessageActions">
            <span>此消息之后的所有内容将被删除 · Ctrl+Enter 提交 / Everything after this message will be removed</span>
            <button className="secondaryButton" type="button" onClick={() => setEditing(false)}>取消 / Cancel</button>
            <button className="primaryButton" type="submit" disabled={!draft.trim()}>重新生成 / Regenerate</button>
          </div>
        </form>
      ) : (
        <MarkdownText text={body} onPreviewImage={onPreviewImage} />
      )}
      {event.attachments?.length > 0 && <AttachmentList attachments={event.attachments} onPreviewImage={onPreviewImage} />}
      {images.length > 0 && <ImageStrip images={images} onPreviewImage={onPreviewImage} />}
      {type === "assistant" && <ReferenceList refs={extractReferences(body)} />}
      {type === "assistant" && event.usage && <UsageLine usage={event.usage} />}
    </article>
  );
}

// Token usage of one turn (all its model requests). The cache figure only shows when the provider reports it.
function UsageLine({ usage }) {
  const format = (value) => Number(value || 0).toLocaleString();
  const hit = usage.cached !== undefined && usage.prompt > 0 ? Math.round((usage.cached / usage.prompt) * 100) : null;
  return (
    <div className="usageLine" title="本轮所有模型请求合计 / Totals over every model request of this turn">
      {hit !== null && <span className={hit >= 70 ? "cacheHit good" : hit >= 30 ? "cacheHit" : "cacheHit low"}>缓存命中 {hit}%</span>}
      <span>输入 {format(usage.prompt)}{hit !== null ? `（缓存 ${format(usage.cached)}）` : ""}</span>
      <span>输出 {format(usage.completion)}</span>
      {usage.requests > 1 && <span>{usage.requests} 次请求</span>}
    </div>
  );
}

function AttachmentList({ attachments, onPreviewImage }) {
  return (
    <div className="attachmentList">
      {attachments.map((item) => {
        const isImage = item.content_type?.startsWith("image/");
        const src = item.url || item.path;
        return (
          <span key={item.path}>
            {isImage ? (
              <button
                className="attachmentThumbButton"
                type="button"
                onClick={() => onPreviewImage?.({ src, alt: item.filename || "upload" })}
                title="View image"
              >
                <img className="attachmentThumb" src={normalizeImageSrc(src)} alt={item.filename || "upload"} />
              </button>
            ) : (
              <Paperclip size={14} />
            )}
            <span>{item.filename || item.path}</span>
            {isImage && (
              <a className="attachmentAction" href={normalizeImageSrc(src)} download={imageDownloadName(item.filename || src)} title="Download image">
                <Download size={13} />
              </a>
            )}
          </span>
        );
      })}
    </div>
  );
}

function ReferenceList({ refs }) {
  if (!refs.length) return null;
  return (
    <div className="referenceList">
      <div>REFERENCE 参考</div>
      {refs.map((ref) => (
        <a key={ref} href={normalizeReferenceHref(ref)} target="_blank" rel="noreferrer">{ref}</a>
      ))}
    </div>
  );
}

function MarkdownText({ text, onPreviewImage }) {
  return (
    <div className="markdownBody">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
          img: ({ src, alt }) => <ImageCard src={src || ""} alt={alt || "image"} onPreviewImage={onPreviewImage} />,
        }}
      >
        {text || ""}
      </ReactMarkdown>
    </div>
  );
}

function ImageStrip({ images, onPreviewImage }) {
  return (
    <div className="imageStrip">
      {images.map((src) => <ImageCard key={src} src={src} alt="tool output" onPreviewImage={onPreviewImage} />)}
    </div>
  );
}

function ImageCard({ src, alt, onPreviewImage }) {
  const href = normalizeImageSrc(src);
  if (!href) return null;
  return (
    <span className="imageCard">
      <button className="imageOpenButton" type="button" onClick={() => onPreviewImage?.({ src, alt })} title="View image">
        <img src={href} alt={alt || "image"} loading="lazy" />
      </button>
      <span className="imageCardActions">
        <button type="button" onClick={() => onPreviewImage?.({ src, alt })}>View</button>
        <a href={href} download={imageDownloadName(src)} target="_blank" rel="noreferrer">
          <Download size={13} />
          <span>Download</span>
        </a>
      </span>
    </span>
  );
}

function ImagePreview({ image, onClose }) {
  useEffect(() => {
    if (!image) return undefined;
    function closeOnEscape(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [image, onClose]);

  if (!image) return null;
  const href = normalizeImageSrc(image.src);
  return (
    <div className="imagePreviewOverlay" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="imagePreview" onClick={(event) => event.stopPropagation()}>
        <div className="imagePreviewBar">
          <span>{image.alt || imageDownloadName(image.src)}</span>
          <div>
            <a className="secondaryButton" href={href} download={imageDownloadName(image.src)} target="_blank" rel="noreferrer">
              <Download size={15} />
              <span>Download</span>
            </a>
            <button className="iconButton neutral" type="button" onClick={onClose} title="Close">
              <X size={16} />
            </button>
          </div>
        </div>
        <img src={href} alt={image.alt || "image preview"} />
      </div>
    </div>
  );
}

function DataView({ text, models }) {
  const filesPane = usePane("data-files", { size: 290, min: 200, max: 480, side: "start" });
  const importPane = usePane("data-import", { size: 320, min: 240, max: 520, side: "end" });
  const [kind, setKind] = useState("instruction");
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState("");
  const [selectedMeta, setSelectedMeta] = useState({ writable: true, scope: "user" });
  const [content, setContent] = useState("");
  const [preview, setPreview] = useState(false);
  const [status, setStatus] = useState("");
  const [importName, setImportName] = useState("");
  const [importContent, setImportContent] = useState("# New Document\n\n");
  const [splitMode, setSplitMode] = useState("simple");
  const [chunkModel, setChunkModel] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ragRefresh, setRagRefresh] = useState({ phase: "idle", message: "" });
  const [prompts, setPrompts] = useState([]);
  const [copied, setCopied] = useState(false);
  const knowledgeUploadRef = useRef(null);
  const isPrompts = kind === "prompts";

  useEffect(() => {
    refreshResourceList();
  }, [kind]);

  useEffect(() => {
    fetchJson("/api/settings").then((data) => {
      const ingestion = data.settings?.rag_ingestion || {};
      setSplitMode(ingestion.split_mode || "simple");
      setChunkModel(ingestion.model || "");
    }).catch(() => {});
  }, []);

  async function updateIngestionSettings(nextMode, nextModel = chunkModel) {
    setSplitMode(nextMode);
    setChunkModel(nextModel);
    await fetchJson("/api/settings", {
      method: "PATCH",
      body: { patch: { rag_ingestion: { split_mode: nextMode, model: nextModel } } },
    });
  }

  async function refreshResourceList() {
    setStatus("");
    setPreview(false);
    if (kind === "prompts") {
      try {
        const data = await fetchJson("/api/prompts");
        const items = data.items || [];
        setPrompts(items);
        setFiles([]);
        if (items.length) selectPrompt(items[0]);
      } catch (error) {
        setStatus(`${text("加载失败", "Load failed")}: ${String(error.message || error)}`);
      }
      return;
    }
    if (kind === "instruction") {
      await openInstruction();
      setFiles([{ path: "data/instruction.md", name: "instruction.md", scope: "user", writable: true }]);
      return;
    }
    const data = await fetchJson(`/api/data/files?kind=${kind}`);
    setFiles(data.items || (data.files || []).map((path) => ({ path, name: path.split("/").pop(), scope: "user", writable: true })));
    setSelected("");
    setContent("");
    setSelectedMeta({ writable: true, scope: "user" });
  }

  function selectPrompt(item) {
    setSelected(item.id);
    setContent(item.content || "");
    setSelectedMeta({ writable: false, scope: "system" });
    setCopied(false);
  }

  async function copyContent() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setStatus(text("复制失败，请手动选择文本。", "Copy failed; select the text by hand."));
    }
  }

  async function openInstruction() {
    const data = await fetchJson("/api/instruction");
    setSelected(data.path || "data/instruction.md");
    setContent(data.content || "");
    setSelectedMeta({ writable: true, scope: "user" });
  }

  async function openFile(item) {
    if (kind === "instruction") {
      await openInstruction();
      return;
    }
    const path = typeof item === "string" ? item : item.path;
    const data = await fetchJson(`/api/data/file?path=${encodeURIComponent(path)}`);
    setSelected(data.path || path);
    setContent(data.content || "");
    setSelectedMeta({ writable: Boolean(data.writable), scope: data.scope || item.scope || "user" });
  }

  async function saveFile() {
    if (!selected || !selectedMeta.writable) return;
    if (kind === "instruction") {
      const data = await fetchJson("/api/instruction", { method: "PUT", body: { content } });
      setContent(data.content || content);
      setStatus("已保存");
    } else {
      const data = await fetchJson("/api/data/file", {
        method: "PUT",
        body: { path: selected, content, split_mode: splitMode, chunk_model: chunkModel },
      });
      setStatus(formatRagRefreshStatus("已保存", data.rag));
    }
    if (kind !== "instruction") await refreshListOnly();
  }

  async function renameSelected() {
    if (!selected || !selectedMeta.writable || !selected.toLowerCase().endsWith(".md")) return;
    const nextName = window.prompt(text("重命名 Markdown 文件", "Rename Markdown file"), selected.split("/").pop() || "");
    if (!nextName) return;
    const data = await fetchJson("/api/data/file/rename", {
      method: "POST",
      body: { path: selected, new_name: nextName, split_mode: splitMode, chunk_model: chunkModel },
    });
    await refreshListOnly();
    await openFile(data.path);
    setStatus(formatRagRefreshStatus(`已重命名：${data.path}`, data.rag));
  }

  async function deleteSelected() {
    if (!selected || !selectedMeta.writable || kind === "instruction") return;
    if (!window.confirm(text("确认删除这个用户文件？", "Delete this user file?"))) return;
    const query = new URLSearchParams({ path: selected, split_mode: splitMode, chunk_model: chunkModel });
    const data = await fetchJson(`/api/data/file?${query.toString()}`, { method: "DELETE" });
    setSelected("");
    setContent("");
    await refreshListOnly();
    setStatus(formatRagRefreshStatus("已删除", data.rag));
  }

  async function refreshListOnly() {
    const data = await fetchJson(`/api/data/files?kind=${kind}`);
    setFiles(data.items || []);
  }

  async function importResource(event) {
    event.preventDefault();
    if (kind !== "instruction" && !importName.trim()) return;
    if (kind === "instruction") {
      const data = await fetchJson("/api/instruction", { method: "PUT", body: { content: importContent } });
      setContent(data.content || importContent);
      setSelected(data.path);
      setStatus("已导入指令");
      return;
    }
    const data = await fetchJson("/api/data/import", {
      method: "POST",
      body: { kind, name: importName, content: importContent, split_mode: splitMode, chunk_model: chunkModel },
    });
    await refreshListOnly();
    await openFile(data.path);
    setImportName("");
    setStatus(formatRagRefreshStatus(`已导入：${data.path}`, data.rag));
  }

  async function reindexRag() {
    if (ragRefresh.phase === "loading") return;
    setRagRefresh({
      phase: "loading",
      message: text("正在刷新 RAG 索引…", "Refreshing RAG index…"),
    });
    try {
      const data = await fetchJson("/api/rag/reindex", {
        method: "POST",
        body: { split_mode: splitMode, chunk_model: chunkModel },
      });
      setRagRefresh({
        phase: "success",
        message: formatRagRefreshStatus(text("RAG 刷新成功", "RAG refresh succeeded"), data),
      });
    } catch (error) {
      setRagRefresh({
        phase: "error",
        message: `${text("RAG 刷新失败", "RAG refresh failed")}: ${String(error.message || error)}`,
      });
    }
  }

  async function ingestKnowledgeFile(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setIngesting(true);
    setStatus(text("正在读取并录入文档…", "Reading and ingesting document…"));
    try {
      const uploadedResponse = await fetch(`${API_BASE}/api/uploads?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      });
      if (!uploadedResponse.ok) throw new Error(`HTTP ${uploadedResponse.status}: ${await uploadedResponse.text()}`);
      const uploaded = await uploadedResponse.json();
      const name = `${file.name.replace(/\.[^.]+$/, "") || "document"}.md`;
      const data = await fetchJson("/api/rag/ingest", {
        method: "POST",
        body: { path: uploaded.path, name, split_mode: splitMode, chunk_model: chunkModel },
      });
      setKind("knowledge");
      const listed = await fetchJson("/api/data/files?kind=knowledge");
      setFiles(listed.items || []);
      const opened = await fetchJson(`/api/data/file?path=${encodeURIComponent(data.path)}`);
      setSelected(opened.path || data.path);
      setContent(opened.content || "");
      setSelectedMeta({ writable: Boolean(opened.writable), scope: opened.scope || "user" });
      setStatus(formatRagRefreshStatus(`已通过 fileReader 录入：${data.path}`, data.index));
    } catch (error) {
      setStatus(`${text("录入失败", "Ingestion failed")}: ${error.message}`);
    } finally {
      setIngesting(false);
    }
  }

  const resourceTabs = [
    ["instruction", text("指令", "INSTRUCTION")],
    ["memory", text("记忆", "MEMORY")],
    ["skills", text("技能", "SKILL")],
    ["knowledge", text("知识", "KNOWLEDGE")],
    ["plans", text("计划", "PLANS")],
    ["prompts", text("提示词", "PROMPTS")],
  ];
  const isMarkdown = selected.toLowerCase().endsWith(".md");
  const promptGroups = [
    ["system", text("系统提示词", "System prompt")],
    ["turn", text("每轮消息", "Per turn")],
    ["other", text("其他模型调用", "Other model calls")],
    ["tools", text("工具定义", "Tool schemas")],
  ];
  const selectedPrompt = isPrompts ? prompts.find((item) => item.id === selected) : null;
  const headerTitle = selectedPrompt ? text(selectedPrompt.title_zh, selectedPrompt.title) : selected || text("资源文件", "Resource file");
  const headerHint = isPrompts
    ? `${text("只读：应用实际发送给模型的提示词，仅用于调试，不能在这里修改。", "Read-only: the prompts the application really sends to a model, for debugging. They cannot be changed here.")}${selectedPrompt ? ` ${text(selectedPrompt.note_zh, selectedPrompt.note)}` : ""}`
    : text(
        `${selectedMeta.writable ? "可编辑用户文件" : "系统级文件只读"}。指令是短规则，记忆是事实，技能是流程，知识是参考文档。`,
        `${selectedMeta.writable ? "Editable user file" : "System file is read-only"}. Instructions are short rules, memory is facts, skills are workflows, and knowledge is reference material.`
      );

  return (
    <section className="dataLayout" style={{ gridTemplateColumns: isPrompts ? splitColumns(filesPane, "r", null) : splitColumns(filesPane, "r", null, "r", importPane) }}>
      <aside className="fileListPane" aria-hidden={filesPane.collapsed || undefined}>
        <div className="segmented vertical">
          {resourceTabs.map(([value, tabLabel]) => (
            <button key={value} className={kind === value ? "active" : ""} onClick={() => setKind(value)} type="button">{tabLabel}</button>
          ))}
        </div>
        {!isPrompts && (
        <section className="dataSidebarSection" aria-label={text("RAG 索引设置", "RAG index settings")}>
          <div className="dataSidebarHeading">
            <Database size={15} />
            <span>{text("RAG 索引", "RAG INDEX")}</span>
          </div>
          <label className="ragField">
            <span>{text("切分方式", "Chunking mode")}</span>
            <div className="segmented ragModeToggle">
              <button className={splitMode === "simple" ? "active" : ""} disabled={ragRefresh.phase === "loading"} onClick={() => updateIngestionSettings("simple")} type="button">{text("简单切分", "Simple")}</button>
              <button className={splitMode === "llm" ? "active" : ""} disabled={ragRefresh.phase === "loading"} onClick={() => updateIngestionSettings("llm")} type="button">{text("LLM 切分", "LLM")}</button>
            </div>
          </label>
          {splitMode === "llm" && (
            <label className="ragField">
              <span>{text("切分模型", "Chunking model")}</span>
              <select value={chunkModel} disabled={ragRefresh.phase === "loading"} onChange={(event) => updateIngestionSettings("llm", event.target.value)}>
                <option value="">{text("默认模型", "Default model")}</option>
                {(models || []).map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
              </select>
            </label>
          )}
          <p className="ragHint">{splitMode === "llm" ? text("短文件免调用；超大文件自动回退，未变化内容复用缓存。", "Short files skip LLM; oversized files fall back; unchanged content is cached.") : text("本地段落切分，不消耗 LLM token。", "Local paragraph splitting uses no LLM tokens.")}</p>
          <div className="ragActions">
            <button className="secondaryButton" onClick={reindexRag} disabled={ragRefresh.phase === "loading"} type="button">
              {ragRefresh.phase === "loading" ? <span className="spinner" /> : <RefreshCw size={16} />}
              <span>{ragRefresh.phase === "loading" ? text("正在刷新…", "Refreshing…") : text("刷新 RAG", "Refresh RAG")}</span>
            </button>
            <button className="secondaryButton" onClick={() => knowledgeUploadRef.current?.click()} disabled={ingesting || ragRefresh.phase === "loading"} type="button"><Paperclip size={16} /><span>{ingesting ? text("正在录入…", "Ingesting…") : text("上传文档", "Upload document")}</span></button>
          </div>
          {ragRefresh.phase !== "idle" && (
            <div className={`ragRefreshFeedback ${ragRefresh.phase}`} role={ragRefresh.phase === "error" ? "alert" : "status"}>
              {ragRefresh.phase === "loading" && <span className="spinner" />}
              {ragRefresh.phase === "success" && <Check size={15} />}
              {ragRefresh.phase === "error" && <X size={15} />}
              <span>{ragRefresh.message}</span>
            </div>
          )}
        </section>
        )}
        <input ref={knowledgeUploadRef} className="hiddenInput" type="file" onChange={ingestKnowledgeFile} />
        <div className="fileList">
          {isPrompts && promptGroups.map(([group, groupLabel]) => (
            <React.Fragment key={group}>
              <div className="fileGroup">{groupLabel}</div>
              {prompts.filter((item) => item.group === group).map((item) => (
                <button key={item.id} className={selected === item.id ? "fileItem active" : "fileItem"} onClick={() => selectPrompt(item)} type="button">
                  <FileText size={15} />
                  <span>{text(item.title_zh, item.title)}</span>
                  <small>{item.content.length.toLocaleString()}</small>
                </button>
              ))}
            </React.Fragment>
          ))}
          {files.map((file) => (
            <button key={file.path} className={selected === file.path ? "fileItem active" : "fileItem"} onClick={() => openFile(file)} type="button">
              <FileText size={15} />
              <span>{file.path}</span>
              <small>{file.scope === "system" ? "SYSTEM" : "USER"}</small>
            </button>
          ))}
        </div>
      </aside>
      <Resizer pane={filesPane} label={text("调整文件栏宽度", "Resize file list")} />
      <div className="editorPane">
        <div className="editorHeader">
          <div>
            <h2>{headerTitle}</h2>
            <span>{headerHint}</span>
          </div>
          {isPrompts ? (
            <div className="rowActions">
              <button className="secondaryButton" onClick={copyContent} disabled={!selected} type="button">{copied ? <Check size={16} /> : <Copy size={16} />}<span>{copied ? text("已复制", "Copied") : text("复制", "Copy")}</span></button>
            </div>
          ) : (
            <div className="rowActions">
              <button className="secondaryButton" onClick={() => setPreview((value) => !value)} disabled={!isMarkdown} type="button"><FileText size={16} /><span>{preview ? text("编辑", "Edit") : text("显示 MD", "Show MD")}</span></button>
              <button className="secondaryButton" onClick={renameSelected} disabled={!selectedMeta.writable || !isMarkdown} type="button"><AtSign size={16} /><span>{text("重命名", "Rename")}</span></button>
              <button className="iconButton danger" onClick={deleteSelected} disabled={!selected || !selectedMeta.writable || kind === "instruction"} type="button" title={text("删除", "Delete")}><Trash2 size={16} /></button>
              <button className="primaryButton" onClick={saveFile} disabled={!selected || !selectedMeta.writable} type="button"><Save size={16} /><span>{text("保存", "Save")}</span></button>
            </div>
          )}
        </div>
        {preview ? (
          <div className="markdownPreview"><MarkdownText text={content} /></div>
        ) : (
          <textarea className="codeEditor" value={content} onChange={(event) => setContent(event.target.value)} readOnly={!selectedMeta.writable} placeholder={text("选择并编辑指令、记忆、技能或知识文件。", "Choose and edit an instruction, memory, skill, or knowledge file.")} />
        )}
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
      {!isPrompts && <Resizer pane={importPane} label={text("调整导入栏宽度", "Resize import panel")} />}
      {!isPrompts && (
        <form className="importPane" onSubmit={importResource} aria-hidden={importPane.collapsed || undefined}>
          <h2>{text("导入", "IMPORT")}</h2>
          <label><span>{text("名称", "Name")}</span><input value={importName} onChange={(event) => setImportName(event.target.value)} placeholder={kind === "skills" ? "my-skill" : kind === "plans" ? "my-plan.md" : "notes.md"} /></label>
          <label><span>{text("内容", "Content")}</span><textarea value={importContent} onChange={(event) => setImportContent(event.target.value)} /></label>
          <button className="primaryButton importSubmit" type="submit"><Plus size={16} /><span>{text("导入到当前分类", "Import to current category")}</span></button>
        </form>
      )}
    </section>
  );
}

function formatRagRefreshStatus(prefix, ragStatus) {
  if (!ragStatus) return prefix;
  const details = [`${ragStatus.chunk_count || 0} 个片段`];
  if (ragStatus.documents_reused) details.push(`复用 ${ragStatus.documents_reused} 个文件`);
  if (ragStatus.llm_calls) details.push(`LLM ${ragStatus.llm_calls} 次 / 约 ${ragStatus.estimated_input_tokens || 0} 输入 token`);
  if (ragStatus.llm_fallbacks) details.push(`回退 ${ragStatus.llm_fallbacks} 次`);
  return `${prefix}：${details.join("，")}`;
}

const PRIORITY_CUT = "__cut__";

// Every "provider:model" value the config offers, in provider order (mirrors backend list_model_items).
function listConfigModels(config) {
  const values = [];
  for (const [provider, entry] of Object.entries(config?.providers || {})) {
    for (const item of Array.isArray(entry?.models) ? entry.models : []) {
      const id = typeof item === "string" ? item : item?.id;
      if (id) values.push(`${provider}:${id}`);
    }
  }
  return values;
}

// Saved order (existing models only) + any new models below the cut line. With nothing saved,
// the current default model is the only one above the line.
function buildPriorityItems(config) {
  const all = listConfigModels(config);
  const saved = [...new Set((Array.isArray(config.model_priority) ? config.model_priority : []).filter((value) => all.includes(value)))];
  const hasSaved = saved.length > 0 && Number.isInteger(config.model_priority_cutoff) && config.model_priority_cutoff >= 1;
  let order = saved;
  if (!hasSaved) {
    const current = config.default_provider && config.default_model ? `${config.default_provider}:${config.default_model}` : "";
    order = all.includes(current) ? [current] : [];
  }
  for (const value of all) if (!order.includes(value)) order.push(value);
  const cutoff = hasSaved ? Math.min(config.model_priority_cutoff, order.length) : Math.min(1, order.length);
  return [...order.slice(0, cutoff), PRIORITY_CUT, ...order.slice(cutoff)];
}

function reconcilePriority(items, config) {
  if (!items.includes(PRIORITY_CUT)) return buildPriorityItems(config);
  const all = listConfigModels(config);
  const kept = items.filter((item) => item === PRIORITY_CUT || all.includes(item));
  const missing = all.filter((value) => !kept.includes(value));
  return [...kept, ...missing];
}

function NativeWebField({ provider, update, text }) {
  const current = provider.native_web;
  const kind = typeof current === "string" ? current : current && typeof current === "object" ? "custom" : "";
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setDraft(current && typeof current === "object" ? JSON.stringify(current, null, 2) : '{\n  "body": {},\n  "tools": []\n}');
    setError("");
  }, [kind]);

  function changeKind(value) {
    if (value === "") update(undefined);
    else if (value === "custom") update({ body: {}, tools: [] });
    else update(value);
  }

  function changeDraft(value) {
    setDraft(value);
    try {
      const parsed = JSON.parse(value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("object expected");
      setError("");
      update(parsed);
    } catch (problem) {
      setError(String(problem.message || problem));
    }
  }

  return (
    <div className="fullWidth nativeWeb">
      <label>
        <span>{text("原生联网搜索", "Native web search")}</span>
        <select value={kind} onChange={(event) => changeKind(event.target.value)}>
          <option value="">{text("未配置", "Not configured")}</option>
          <option value="deepseek">{text("DeepSeek（Search / Fetch 接口）", "DeepSeek (Search / Fetch API)")}</option>
          <option value="openrouter">OpenRouter (web plugin)</option>
          <option value="dashscope">{text("阿里云百炼 / Qwen (enable_search)", "Alibaba DashScope / Qwen (enable_search)")}</option>
          <option value="custom">{text("自定义请求参数", "Custom request fields")}</option>
        </select>
      </label>
      {kind === "custom" && (
        <>
          <textarea className="codeEditor" value={draft} onChange={(event) => changeDraft(event.target.value)} spellCheck={false} />
          <small>{text('body 会合并进每次请求；tools 会追加到请求的 tools。例如 {"body":{"enable_search":true}}。', 'body is merged into every request; tools are appended to the request tools. Example: {"body":{"enable_search":true}}.')}</small>
          {error && <small className="webModeWarn">{error}</small>}
        </>
      )}
      {kind === "" && String(provider.base_url || "").includes("api.deepseek.com") && (
        <small>{text("已自动识别为 DeepSeek 官方接口：原生模式下提供 nativeWebSearch 与 webFetch 工具，无需配置。", "Recognised as the official DeepSeek API: Native mode offers the nativeWebSearch and webFetch tools, no setup needed.")}</small>
      )}
      {kind !== "" && kind !== "custom" && <small>{text("在聊天设置里把“联网”调到“原生”或“两者”后生效。", "Takes effect when Web is set to Native or Both in the chat settings.")}</small>}
    </div>
  );
}

function ConfigView({ onSaved, text }) {
  const listPane = usePane("config-list", { size: 280, min: 200, max: 460, side: "start" });
  const [config, setConfig] = useState({ providers: {} });
  const [selectedProvider, setSelectedProvider] = useState("");
  const [status, setStatus] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [newModel, setNewModel] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [view, setView] = useState("provider");
  const [priorityItems, setPriorityItems] = useState([]);
  const [dragIndex, setDragIndex] = useState(null);
  const [dropIndex, setDropIndex] = useState(null);

  useEffect(() => {
    refreshConfig();
  }, []);

  // Keep the priority list in step with the models that exist: drop removed ones, append new ones below the cut.
  const modelKey = listConfigModels(config).join("|");
  useEffect(() => {
    setPriorityItems((current) => reconcilePriority(current, config));
  }, [modelKey]);

  async function refreshConfig(preferredProvider = "") {
    const data = await fetchJson("/api/config");
    const nextConfig = data.config || { providers: {} };
    setConfig(nextConfig);
    setPriorityItems(buildPriorityItems(nextConfig));
    setSelectedProvider(preferredProvider || nextConfig.default_provider || Object.keys(nextConfig.providers || {})[0] || "");
    setApiKey("");
    setStatus("");
  }

  function updateConfig(updater) {
    setConfig((current) => updater(structuredClone(current)));
  }

  function updateProvider(key, value) {
    updateConfig((draft) => {
      draft.providers ||= {};
      draft.providers[selectedProvider] ||= {};
      draft.providers[selectedProvider][key] = value;
      return draft;
    });
  }

  function addProvider() {
    const name = window.prompt("Provider 名称", "custom");
    if (!name) return;
    updateConfig((draft) => {
      draft.providers ||= {};
      draft.providers[name] ||= { base_url: "", api_key_env: "", models: [] };
      draft.default_provider ||= name;
      return draft;
    });
    setSelectedProvider(name);
    setApiKey("");
  }

  function renameProvider() {
    if (!selectedProvider) return;
    const nextName = window.prompt("重命名 Provider", selectedProvider);
    if (!nextName || nextName === selectedProvider) return;
    updateConfig((draft) => {
      draft.providers ||= {};
      draft.providers[nextName] = draft.providers[selectedProvider] || {};
      delete draft.providers[selectedProvider];
      if (draft.default_provider === selectedProvider) draft.default_provider = nextName;
      return draft;
    });
    setSelectedProvider(nextName);
    setApiKey("");
  }

  function deleteProvider() {
    if (!selectedProvider || !window.confirm(`删除 Provider: ${selectedProvider}？`)) return;
    updateConfig((draft) => {
      draft.providers ||= {};
      delete draft.providers[selectedProvider];
      if (draft.default_provider === selectedProvider) draft.default_provider = Object.keys(draft.providers)[0] || "";
      if (draft.default_model && !draft.default_provider) draft.default_model = "";
      return draft;
    });
    setSelectedProvider("");
  }

  async function saveConfig(showStatus = true) {
    try {
      const payload = structuredClone(config);
      if (selectedProvider && apiKey.trim()) payload.providers[selectedProvider].api_key = apiKey.trim();
      const cutIndex = priorityItems.indexOf(PRIORITY_CUT);
      if (cutIndex >= 0) {
        payload.model_priority = priorityItems.filter((item) => item !== PRIORITY_CUT);
        payload.model_priority_cutoff = cutIndex;
      }
      const data = await fetchJson("/api/config", { method: "PUT", body: { config: payload } });
      setConfig(data.config || config);
      if (data.config) setPriorityItems(buildPriorityItems(data.config));
      setApiKey("");
      if (showStatus) setStatus(text("模型 API 配置已保存", "Model API configuration saved"));
      await onSaved?.();
      return true;
    } catch (error) {
      setStatus(`${text("保存失败", "Save failed")}: ${String(error.message || error)}`);
      return false;
    }
  }

  async function refreshModels() {
    if (!selectedProvider || refreshing) return;
    setRefreshing(true);
    try {
      if (!(await saveConfig(false))) return;
      const data = await fetchJson(`/api/config/providers/${encodeURIComponent(selectedProvider)}/models/refresh`, { method: "POST" });
      setConfig(data.config || config);
      setStatus(text(`已刷新 ${data.count || 0} 个模型`, `Refreshed ${data.count || 0} models`));
      await onSaved?.();
    } catch (error) {
      setStatus(`${text("刷新失败", "Refresh failed")}: ${String(error.message || error)}`);
    } finally {
      setRefreshing(false);
    }
  }

  function addManualModel() {
    const value = newModel.trim();
    if (!value) return;
    updateProvider("models", [...(Array.isArray(provider.models) ? provider.models : []), value]);
    setNewModel("");
  }

  function removeModel(modelId) {
    updateProvider("models", (Array.isArray(provider.models) ? provider.models : []).filter((item) => modelValue(item) !== modelId));
  }

  function setDefaultModel(modelId) {
    updateConfig((draft) => ({ ...draft, default_provider: selectedProvider, default_model: modelId }));
    // Make it the head of the fallback chain too, keeping the cut where it was.
    setPriorityItems((items) => {
      const value = `${selectedProvider}:${modelId}`;
      const rest = items.filter((item) => item !== value);
      return [value, ...rest];
    });
  }

  // priorityItems = [...chain, PRIORITY_CUT, ...pool]: the chain is tried in order, the pool never is.
  const cutIndex = priorityItems.indexOf(PRIORITY_CUT);
  const priorityChain = priorityItems.slice(0, Math.max(cutIndex, 0));
  const priorityPool = priorityItems.slice(cutIndex + 1);
  const [poolQuery, setPoolQuery] = useState("");

  function setSplit(chain, pool) {
    setPriorityItems([...chain, PRIORITY_CUT, ...pool]);
  }

  function moveInChain(from, to) {
    if (from === null || to === null || from === to || to < 0 || to >= priorityChain.length) return;
    const next = [...priorityChain];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    setSplit(next, priorityPool);
  }

  function removeFromChain(value) {
    setSplit(priorityChain.filter((item) => item !== value), [value, ...priorityPool]);
  }

  function addToChain(value) {
    setSplit([...priorityChain, value], priorityPool.filter((item) => item !== value));
  }

  const providers = config.providers || {};
  const provider = selectedProvider ? providers[selectedProvider] || {} : {};
  const providerModels = Array.isArray(provider.models) ? provider.models : [];

  return (
    <section className="configLayout" style={{ gridTemplateColumns: splitColumns(listPane, "r", null) }}>
      <aside className="providerPane" aria-hidden={listPane.collapsed || undefined}>
        <div className="editorHeader">
          <h2>{text("供应商", "PROVIDER")}</h2>
          <button className="iconButton neutral" onClick={addProvider} type="button" title={text("新增", "Add")}><Plus size={16} /></button>
        </div>
        <button className={view === "priority" ? "providerItem active priorityEntry" : "providerItem priorityEntry"} onClick={() => setView("priority")} type="button">
          <span><ListOrdered size={15} />{text("默认模型优先级", "Default model priority")}</span>
        </button>
        {Object.keys(providers).map((name) => (
          <button key={name} className={view === "provider" && selectedProvider === name ? "providerItem active" : "providerItem"} onClick={() => { setSelectedProvider(name); setView("provider"); }} type="button">
            <span><Key size={15} />{name}</span>
            {config.default_provider === name && <small>{text("默认", "DEFAULT")}</small>}
          </button>
        ))}
      </aside>
      <Resizer pane={listPane} label={text("调整列表宽度", "Resize list")} />
      {view === "priority" ? (
      <div className="modelEditor priorityEditor">
        <div className="editorHeader">
          <div>
            <h2>{text("默认模型优先级", "DEFAULT MODEL PRIORITY")}</h2>
            <span>{text("默认模型调用失败时，按下面的队列顺序依次换下一个。队列之外的模型不会被自动尝试。", "When the default model fails, the next one in the queue is tried. Models outside the queue are never tried automatically.")}</span>
          </div>
          <div className="rowActions">
            <button className="secondaryButton" type="button" onClick={() => setPriorityItems(buildPriorityItems(config))}><RefreshCw size={16} /><span>{text("重置", "Reset")}</span></button>
            <button className="primaryButton" onClick={() => saveConfig()} type="button"><Save size={16} /><span>{text("保存", "Save")}</span></button>
          </div>
        </div>
        <div className="prioBody">
          <section className="prioCard">
            <header>
              <h3>{text("优先队列", "Priority queue")}</h3>
              <small>{priorityChain.length} {text("个模型 · 从上到下依次尝试", "models · tried top to bottom")}</small>
            </header>
            {priorityChain.length === 0 ? (
              <p className="prioEmpty">{text("队列为空，没有默认模型。从下方“未启用”里加入至少一个模型。", "The queue is empty, so there is no default model. Add at least one from “Not in use” below.")}</p>
            ) : (
              <ol className="prioList">
                {priorityChain.map((value, index) => {
                  const [providerName, ...rest] = value.split(":");
                  return (
                    <li
                      key={value}
                      className={`prioRow${dragIndex === index ? " dragging" : ""}${dropIndex === index && dragIndex !== index ? " dropTarget" : ""}`}
                      draggable
                      onDragStart={(event) => { setDragIndex(index); event.dataTransfer.effectAllowed = "move"; event.dataTransfer.setData("text/plain", value); }}
                      onDragOver={(event) => { event.preventDefault(); setDropIndex(index); }}
                      onDrop={(event) => { event.preventDefault(); moveInChain(dragIndex, index); setDragIndex(null); setDropIndex(null); }}
                      onDragEnd={() => { setDragIndex(null); setDropIndex(null); }}
                    >
                      <GripVertical size={16} className="prioGrip" aria-hidden="true" />
                      <span className={index === 0 ? "prioRank first" : "prioRank"}>{index + 1}</span>
                      <span className="prioName" title={value}>
                        <b>{rest.join(":")}</b>
                        <small>{providerName}</small>
                      </span>
                      <span className="prioTag">{index === 0 ? text("默认", "Default") : text("备用", "Fallback")}</span>
                      <span className="prioButtons">
                        <button type="button" onClick={() => moveInChain(index, index - 1)} disabled={index === 0} title={text("上移", "Move up")} aria-label={text("上移", "Move up")}><ChevronUp size={16} /></button>
                        <button type="button" onClick={() => moveInChain(index, index + 1)} disabled={index === priorityChain.length - 1} title={text("下移", "Move down")} aria-label={text("下移", "Move down")}><ChevronDown size={16} /></button>
                        <button type="button" className="remove" onClick={() => removeFromChain(value)} title={text("移出队列", "Remove from queue")} aria-label={text("移出队列", "Remove from queue")}><X size={16} /></button>
                      </span>
                    </li>
                  );
                })}
              </ol>
            )}
            {priorityChain.length > 0 && (
              <p className="prioNote">
                <Scissors size={13} />
                {text(`“${priorityChain[priorityChain.length - 1]}” 是最后一个：它也失败时，不会再尝试其他模型。`, `“${priorityChain[priorityChain.length - 1]}” is the last one: if it fails too, no other model is tried.`)}
              </p>
            )}
          </section>

          <section className="prioCard muted">
            <header>
              <h3>{text("未启用", "Not in use")}</h3>
              <small>{priorityPool.length} {text("个模型 · 不会被自动尝试", "models · never tried automatically")}</small>
            </header>
            {priorityPool.length > 6 && (
              <input className="prioSearch" value={poolQuery} onChange={(event) => setPoolQuery(event.target.value)} placeholder={text("搜索模型…", "Search models...")} />
            )}
            <ul className="prioList">
              {priorityPool.filter((value) => value.toLowerCase().includes(poolQuery.trim().toLowerCase())).map((value) => {
                const [providerName, ...rest] = value.split(":");
                return (
                  <li className="prioRow pool" key={value}>
                    <span className="prioName" title={value}>
                      <b>{rest.join(":")}</b>
                      <small>{providerName}</small>
                    </span>
                    <button className="prioAdd" type="button" onClick={() => addToChain(value)}>
                      <Plus size={14} />
                      {text("加入队列", "Add")}
                    </button>
                  </li>
                );
              })}
            </ul>
            {priorityPool.length === 0 && <p className="prioEmpty">{text("所有模型都已在队列中。", "Every model is already in the queue.")}</p>}
          </section>
        </div>
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
      ) : (
      <div className="modelEditor">
        <div className="editorHeader">
          <div>
            <h2>{text("模型 API 配置", "MODEL API CONFIG")}</h2>
            <span>{text("选择左侧供应商后填写 Base URL、API Key 或环境变量，以及模型列表。", "Choose a provider, then fill in Base URL, API Key or env var, and model list.")}</span>
          </div>
          <div className="rowActions">
            <button className="secondaryButton" onClick={renameProvider} disabled={!selectedProvider} type="button"><AtSign size={16} /><span>{text("重命名", "Rename")}</span></button>
            <button className="dangerButton" onClick={deleteProvider} disabled={!selectedProvider} type="button"><Trash2 size={16} /><span>{text("删除", "Delete")}</span></button>
            <button className="secondaryButton" onClick={refreshModels} disabled={!selectedProvider || refreshing} type="button"><RefreshCw size={16} /><span>{refreshing ? text("刷新中", "Refreshing") : text("刷新模型", "Refresh models")}</span></button>
            <button className="primaryButton" onClick={() => saveConfig()} type="button"><Save size={16} /><span>{text("保存", "Save")}</span></button>
          </div>
        </div>
        {selectedProvider ? (
          <div className="modelForm">
            <label><span>{text("当前供应商", "Current Provider")}</span><input value={selectedProvider} readOnly /></label>
            <label><span>Base URL</span><input value={provider.base_url || ""} onChange={(event) => updateProvider("base_url", event.target.value)} placeholder="https://api.deepseek.com/v1" /></label>
            <label><span>API Key</span><input type="password" autoComplete="new-password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={provider.has_api_key ? text("已保存，留空则保持不变", "Saved; leave blank to keep") : text("输入 API Key", "Enter API Key")} /></label>
            <label><span>API Key Env</span><input value={provider.api_key_env || ""} onChange={(event) => updateProvider("api_key_env", event.target.value)} placeholder="DEEPSEEK_API_KEY" /></label>
            <NativeWebField provider={provider} update={(value) => updateProvider("native_web", value)} text={text} />
            <label className="fullWidth">
              <span>{text("思考强度参数", "Reasoning parameter")}</span>
              <select value={provider.reasoning || ""} onChange={(event) => updateProvider("reasoning", event.target.value)}>
                <option value="">{text("自动识别（按 Base URL）", "Auto-detect (from Base URL)")}</option>
                <option value="deepseek">DeepSeek: thinking + reasoning_effort</option>
                <option value="openai">OpenAI: reasoning_effort</option>
                <option value="openrouter">OpenRouter: reasoning.effort</option>
                <option value="dashscope">{text("百炼: enable_thinking + thinking_budget", "DashScope: enable_thinking + thinking_budget")}</option>
                <option value="none">{text("不发送思考参数", "Never send a reasoning parameter")}</option>
              </select>
              <small>{text("聊天输入栏里的“思考强度”滑条会按这里的格式发给该供应商；不认识的供应商选“不发送”。请求被拒时会自动去掉该参数重试。", "The thinking-effort slider in the composer is sent to this provider in this format. Use \"never send\" for providers that do not know it; a rejected request is retried without the parameter.")}</small>
            </label>
            <label><span>{text("默认供应商", "Default Provider")}</span><input value={config.default_provider || ""} readOnly /></label>
            <div className="fullWidth modelCatalog">
              <div className="modelCatalogHeader">
                <span>{text("支持的模型", "Supported models")}</span>
                <small>{text("优先使用刷新按钮从 API 获取，也可以手动添加", "Refresh from the API or add one manually")}</small>
              </div>
              <div className="manualModelRow">
                <input value={newModel} onChange={(event) => setNewModel(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); addManualModel(); } }} placeholder="model-id" />
                <button className="secondaryButton" type="button" onClick={addManualModel}><Plus size={16} /><span>{text("添加", "Add")}</span></button>
              </div>
              <div className="modelList">
                {providerModels.map((item) => {
                  const id = modelValue(item);
                  const isDefault = config.default_provider === selectedProvider && config.default_model === id;
                  return (
                    <div className="modelRow" key={id}>
                      <button className={isDefault ? "modelName active" : "modelName"} type="button" onClick={() => setDefaultModel(id)} title={text("设为默认模型", "Set as default model")}>
                        {isDefault && <Check size={14} />}{id}
                      </button>
                      <button className="iconButton neutral" type="button" onClick={() => removeModel(id)} title={text("移除", "Remove")}><X size={15} /></button>
                    </div>
                  );
                })}
                {providerModels.length === 0 && <div className="emptyState">{text("尚未加载模型，请填写 API 后刷新。", "No models loaded. Configure the API and refresh.")}</div>}
              </div>
            </div>
          </div>
        ) : (
          <div className="emptyState">{text("左侧新增或选择一个 Provider。", "Add or choose a provider on the left.")}</div>
        )}
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
      )}
    </section>
  );
}

function CustomToolsView({ text }) {
  const listPane = usePane("tools-list", { size: 280, min: 200, max: 460, side: "start" });
  const emptyTool = {
    name: "",
    description: "",
    parametersText: '{\n  "type": "object",\n  "properties": {},\n  "required": []\n}',
    code: "def run(arguments):\n    return arguments\n",
  };
  const [tools, setTools] = useState([]);
  const [selected, setSelected] = useState("");
  const [form, setForm] = useState(emptyTool);
  const [status, setStatus] = useState("");

  useEffect(() => { refreshTools(); }, []);

  async function refreshTools() {
    const data = await fetchJson("/api/custom-tools");
    setTools(data.tools || []);
  }

  async function openTool(name) {
    const data = await fetchJson(`/api/custom-tools/${encodeURIComponent(name)}`);
    setSelected(name);
    setForm({
      name: data.name || name,
      description: data.description || "",
      parametersText: JSON.stringify(data.parameters || { type: "object", properties: {} }, null, 2),
      code: data.code || "",
    });
    setStatus("");
  }

  async function saveTool() {
    try {
      const parameters = JSON.parse(form.parametersText);
      const name = form.name.trim();
      await fetchJson(`/api/custom-tools/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: { name, description: form.description, parameters, code: form.code },
      });
      setSelected(name);
      setStatus(text("自建工具已保存，可在 Agent 模式中作为 custom__名称 调用。", "Custom tool saved and available as custom__name in Agent mode."));
      await refreshTools();
    } catch (error) {
      setStatus(`${text("保存失败", "Save failed")}: ${String(error.message || error)}`);
    }
  }

  async function deleteTool() {
    if (!selected) return;
    if (!window.confirm(text(`删除自建工具 custom__${selected}？`, `Delete custom tool custom__${selected}?`))) return;
    try {
      await fetchJson(`/api/custom-tools/${encodeURIComponent(selected)}`, { method: "DELETE" });
      setSelected("");
      setForm(emptyTool);
      setStatus(text("已删除自建工具", "Custom tool deleted"));
      await refreshTools();
    } catch (error) {
      setStatus(`${text("删除失败", "Delete failed")}: ${String(error.message || error)}`);
    }
  }

  return (
    <section className="configLayout" style={{ gridTemplateColumns: splitColumns(listPane, "r", null) }}>
      <aside className="providerPane" aria-hidden={listPane.collapsed || undefined}>
        <div className="editorHeader"><h2>{text("自建工具", "CUSTOM TOOLS")}</h2><button className="iconButton neutral" type="button" onClick={() => { setSelected(""); setForm(emptyTool); setStatus(""); }}><Plus size={16} /></button></div>
        {tools.map((item) => <button className={selected === item.name ? "providerItem active" : "providerItem"} key={item.name} type="button" onClick={() => openTool(item.name)}><span><Wrench size={15} />custom__{item.name}</span></button>)}
      </aside>
      <Resizer pane={listPane} label={text("调整列表宽度", "Resize list")} />
      <div className="modelEditor">
        <div className="editorHeader"><div><h2>{text("工具代码", "TOOL CODE")}</h2><span>{text("用户级工具与 backend/tools 系统工具分开保存。代码必须定义 run(arguments)。", "User tools are separate from system tools and must define run(arguments).")}</span></div><div className="rowActions"><button className="dangerButton" type="button" onClick={deleteTool} disabled={!selected}><Trash2 size={16} /><span>{text("删除", "Delete")}</span></button><button className="primaryButton" type="button" onClick={saveTool}><Save size={16} /><span>{text("保存", "Save")}</span></button></div></div>
        <div className="customToolForm">
          <label><span>{text("名称", "Name")}</span><input value={form.name} onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))} placeholder="my_tool" /></label>
          <label><span>{text("描述", "Description")}</span><input value={form.description} onChange={(event) => setForm((current) => ({ ...current, description: event.target.value }))} /></label>
          <label><span>Parameters JSON Schema</span><textarea value={form.parametersText} onChange={(event) => setForm((current) => ({ ...current, parametersText: event.target.value }))} /></label>
          <label><span>Python</span><textarea className="codeEditor" value={form.code} onChange={(event) => setForm((current) => ({ ...current, code: event.target.value }))} /></label>
        </div>
        {status && <p className="statusLine">{status}</p>}
      </div>
    </section>
  );
}

function SystemView({ theme, setTheme, language, setLanguage, text }) {
  const rows = [
    {
      icon: <Sun size={15} />,
      title: text("主题", "Theme"),
      hint: text("浅色、深色，或跟随操作系统。", "Light, dark, or follow the operating system."),
      value: theme,
      onChange: setTheme,
      options: [
        { value: "system", label: text("跟随系统", "System") },
        { value: "light", label: text("浅色", "Light") },
        { value: "dark", label: text("深色", "Dark") },
      ],
    },
    {
      icon: <Languages size={15} />,
      title: text("界面语言", "Language"),
      hint: text("“中英双语”会把每个文字显示成“中文 / English”。", "“Both” shows every label as “中文 / English”."),
      value: language,
      onChange: setLanguage,
      options: [
        { value: "zh", label: "中文" },
        { value: "en", label: "English" },
        { value: "both", label: text("中英双语", "Both") },
      ],
    },
  ];
  return (
    <section className="systemLayout">
      <div className="systemPanel">
        <div className="editorHeader">
          <div>
            <h2>{text("系统设置", "System settings")}</h2>
            <span>{text("这些设置只影响界面，不会混入当前对话提示词。", "These settings only affect the UI and are not injected into chat prompts.")}</span>
          </div>
        </div>
        <div className="settingRows">
          {rows.map((row) => (
            <div className="settingRow" key={row.title}>
              <div className="settingText">
                <strong>{row.icon}{row.title}</strong>
                <span>{row.hint}</span>
              </div>
              <Segmented options={row.options} value={row.value} onChange={row.onChange} ariaLabel={row.title} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function AutomationView({ options, setOptions, text }) {
  const listPane = usePane("automation-list", { size: 280, min: 200, max: 460, side: "start" });
  const emptyForm = {
    title: "",
    action: "reminder",
    enabled: true,
    prompt: "",
    code: "",
    mcp_server: "",
    mcp_tool: "",
    scheduleText: '{\n  "kind": "once",\n  "nextRunAt": ""\n}',
    mcpArgumentsText: "{}",
    mcpConfigText: "{}",
  };
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState("");
  const [selectedItem, setSelectedItem] = useState(null);
  const [form, setForm] = useState(emptyForm);
  const [runs, setRuns] = useState([]);
  const [status, setStatus] = useState("");

  useEffect(() => {
    refreshAutomations();
  }, []);

  async function refreshAutomations() {
    try {
      const data = await fetchJson("/api/automations");
      setItems(data.items || []);
      if (selected) {
        const detail = await fetchJson(`/api/automations/${encodeURIComponent(selected)}`).catch(() => null);
        if (detail) setRuns(detail.runs || []);
      }
    } catch (error) {
      setStatus(`${text("加载失败", "Load failed")}: ${String(error.message || error)}`);
    }
  }

  async function openAutomation(id) {
    const data = await fetchJson(`/api/automations/${encodeURIComponent(id)}`);
    const content = data.content || {};
    setSelected(id);
    setSelectedItem(data.item || null);
    setRuns(data.runs || []);
    setForm({
      title: content.title || "",
      action: content.action || "reminder",
      enabled: content.enabled ?? true,
      prompt: content.prompt || "",
      code: content.code || "",
      mcp_server: content.mcp_server || "",
      mcp_tool: content.mcp_tool || "",
      scheduleText: JSON.stringify(content.schedule || {}, null, 2),
      mcpArgumentsText: JSON.stringify(content.mcp_arguments || {}, null, 2),
      mcpConfigText: JSON.stringify(content.mcp_config || {}, null, 2),
    });
    setStatus("");
  }

  function newAutomation() {
    setSelected("");
    setSelectedItem(null);
    setForm(emptyForm);
    setRuns([]);
    setStatus("");
  }

  async function saveAutomation(event) {
    event.preventDefault();
    let payload;
    try {
      payload = automationPayloadFromForm(form);
    } catch (error) {
      setStatus(`${text("JSON 格式错误", "Invalid JSON")}: ${String(error.message || error)}`);
      return;
    }
    const url = selected ? `/api/automations/${encodeURIComponent(selected)}` : "/api/automations";
    const method = selected ? "PUT" : "POST";
    try {
      const data = await fetchJson(url, { method, body: payload });
      setSelected(data.item?.id || selected);
      setSelectedItem(data.item || null);
      setRuns(data.item?.recent_runs || []);
      await refreshAutomations();
      setStatus(text("已保存自动化", "Automation saved"));
    } catch (error) {
      setStatus(`${text("保存失败", "Save failed")}: ${String(error.message || error)}`);
    }
  }

  async function deleteAutomation() {
    if (!selected || !window.confirm(text("删除这个自动化？", "Delete this automation?"))) return;
    await fetchJson(`/api/automations/${encodeURIComponent(selected)}`, { method: "DELETE" });
    newAutomation();
    await refreshAutomations();
    setStatus(text("已删除自动化", "Automation deleted"));
  }

  return (
    <section className="configLayout" style={{ gridTemplateColumns: splitColumns(listPane, "r", null) }}>
      <aside className="providerPane" aria-hidden={listPane.collapsed || undefined}>
        <div className="editorHeader">
          <h2>{text("自动化", "AUTOMATION")}</h2>
          <button className="iconButton neutral" onClick={newAutomation} type="button" title={text("新增", "Add")}><Plus size={16} /></button>
        </div>
        <label>
          <span>{text("模型自动设置", "Model Auto Setup")}</span>
          <select value={options.automation_mode} onChange={(event) => setOptions((current) => ({ ...current, automation_mode: event.target.value }))}>
            <option value="off">{text("关闭", "Off")}</option>
            <option value="auto">{text("允许", "Allowed")}</option>
          </select>
          <small>{text("开启后，模型可以通过 automation tool 保存提醒、LLM 步骤、脚本或 MCP 自动化。", "When enabled, the model can use the automation tool to save reminders, LLM steps, scripts, or MCP automations.")}</small>
        </label>
        <div className="fileList">
          {items.map((item) => (
            <button key={item.id} className={selected === item.id ? "fileItem active" : "fileItem"} onClick={() => openAutomation(item.id)} type="button">
              <RefreshCw size={15} />
              <span>
                {item.title || item.id}
                <small className="inlineMeta">{scheduleSummary(item, text)}</small>
              </span>
              <small>{item.enabled ? item.action : text("关闭", "OFF")}</small>
            </button>
          ))}
        </div>
      </aside>
      <Resizer pane={listPane} label={text("调整列表宽度", "Resize list")} />
      <form className="modelEditor" onSubmit={saveAutomation}>
        <div className="editorHeader">
          <div>
            <h2>{selected ? form.title || selected : text("新增自动化", "New Automation")}</h2>
            <span>{text("人工创建会保存到同一个自动化目录；模型自动创建也会出现在这里。", "Manual entries are saved to the same automation directory; model-created entries appear here too.")}</span>
          </div>
          <div className="rowActions">
            <button className="dangerButton" onClick={deleteAutomation} disabled={!selected} type="button"><Trash2 size={16} /><span>{text("删除", "Delete")}</span></button>
            <button className="secondaryButton" onClick={refreshAutomations} type="button"><RefreshCw size={16} /><span>{text("刷新", "Refresh")}</span></button>
            <button className="primaryButton" type="submit"><Save size={16} /><span>{text("保存", "Save")}</span></button>
          </div>
        </div>
        <div className="modelForm">
          <label><span>{text("标题", "Title")}</span><input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} placeholder={text("检查报告", "Check report")} /></label>
          <SelectField label={text("类型", "Action")} value={form.action} onChange={(value) => setForm({ ...form, action: value })} values={choices("automation", text)} />
          <label className="checkLine"><input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} /><span>{text("启用", "Enabled")}</span></label>
          <div className="automationMeta fullWidth">
            <div><strong>{text("时间类型", "Schedule Type")}</strong><span>{safeScheduleValue(form.scheduleText, "kind") || "once"}</span></div>
            <div><strong>{text("下次执行", "Next Run")}</strong><span>{safeScheduleValue(form.scheduleText, "nextRunAt") || text("未设置", "Not set")}</span></div>
            <div><strong>{text("操作记录", "Run Log")}</strong><code>backend/runtime/automation_runs/runs-YYYYMMDD.jsonl</code></div>
            <div><strong>{text("自动化对话", "Automation Chat")}</strong><span>{selectedItem?.conversation_id || text("首次执行后创建", "Created on first run")}</span></div>
          </div>
          <label className="fullWidth"><span>{text("提示词", "Prompt")}</span><textarea value={form.prompt} onChange={(event) => setForm({ ...form, prompt: event.target.value })} placeholder={text("到时间后要提醒或交给模型执行的内容", "Prompt to remind or hand to the model when due")} /></label>
          <label className="fullWidth"><span>{text("计划 JSON", "Schedule JSON")}</span><textarea value={form.scheduleText} onChange={(event) => setForm({ ...form, scheduleText: event.target.value })} /></label>
          {form.action === "script" && (
            <label className="fullWidth"><span>Python</span><textarea value={form.code} onChange={(event) => setForm({ ...form, code: event.target.value })} placeholder="print('hello')" /></label>
          )}
          {form.action === "mcp" && (
            <>
              <label><span>{text("MCP 服务器", "MCP Server")}</span><input value={form.mcp_server} onChange={(event) => setForm({ ...form, mcp_server: event.target.value })} /></label>
              <label><span>{text("MCP 工具", "MCP Tool")}</span><input value={form.mcp_tool} onChange={(event) => setForm({ ...form, mcp_tool: event.target.value })} /></label>
              <label className="fullWidth"><span>{text("MCP 参数 JSON", "MCP Arguments JSON")}</span><textarea value={form.mcpArgumentsText} onChange={(event) => setForm({ ...form, mcpArgumentsText: event.target.value })} /></label>
            </>
          )}
          {form.action === "configureMcp" && (
            <label className="fullWidth"><span>{text("MCP 配置 JSON", "MCP Config JSON")}</span><textarea value={form.mcpConfigText} onChange={(event) => setForm({ ...form, mcpConfigText: event.target.value })} /></label>
          )}
        </div>
        <div className="runList">
          <div className="labelText">{text("最近运行记录", "Recent Runs")}</div>
          {runs.length === 0 ? (
            <div className="emptyState smallEmpty">{text("还没有运行记录。", "No run records yet.")}</div>
          ) : (
            runs.map((run) => (
              <article className={run.status === "error" ? "runItem error" : "runItem"} key={run.run_id || run.started_at}>
                <div>
                  <strong>{run.status || "ok"}</strong>
                  <span>{run.started_at}</span>
                  <code>{run.path}</code>
                </div>
                <pre>{formatRunRecord(run)}</pre>
              </article>
            ))
          )}
        </div>
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </form>
    </section>
  );
}

function McpView({ text }) {
  const formPane = usePane("mcp-form", { size: 420, min: 300, max: 640, side: "end" });
  const [config, setConfig] = useState({ servers: {} });
  const [form, setForm] = useState({
    name: "",
    enabled: true,
    transport: "streamable_http",
    url: "",
    headerRows: [{ key: "Authorization", value: "Bearer " }],
    command: "",
    argsText: "",
    envRows: [{ key: "", value: "" }],
    timeout: 5,
    sse_read_timeout: 300,
  });
  const [status, setStatus] = useState("");

  useEffect(() => { refresh(); }, []);

  async function refresh() {
    try {
      const data = await fetchJson("/api/mcp/servers");
      setConfig(data || { servers: {} });
    } catch (error) {
      setStatus(`${text("加载失败", "Load failed")}: ${String(error.message || error)}`);
    }
  }

  function editServer(name, server) {
    setForm({
      name,
      enabled: server.enabled ?? true,
      transport: server.transport || "streamable_http",
      url: server.url || "",
      headerRows: headerRowsFromObject(server.headers || {}),
      command: server.command || "",
      argsText: Array.isArray(server.args) ? server.args.join("\n") : "",
      envRows: headerRowsFromObject(server.env || {}),
      timeout: server.timeout || 5,
      sse_read_timeout: server.sse_read_timeout || 300,
    });
  }

  async function saveServer(event) {
    event.preventDefault();
    const payload = buildMcpPayload(form);
    if (!payload.name) {
      setStatus(text("名称不能为空", "Name is required"));
      return;
    }
    try {
      const data = await fetchJson(`/api/mcp/servers/${encodeURIComponent(payload.name)}`, { method: "PUT", body: payload });
      setConfig(data);
      const saved = data.servers?.[payload.name] || payload;
      editServer(payload.name, saved);
      setStatus(`${text("已保存", "Saved")} ${payload.name}`);
    } catch (error) {
      setStatus(`${text("保存失败", "Save failed")}: ${String(error.message || error)}`);
    }
  }

  async function testCurrentServer() {
    const payload = buildMcpPayload(form);
    if (payload.transport === "streamable_http" && !payload.url) {
      setStatus(text("测试失败：需要 URL", "Test failed: url is required"));
      return;
    }
    if (payload.transport === "stdio" && !payload.command) {
      setStatus(text("测试失败：需要命令", "Test failed: command is required"));
      return;
    }
    try {
      const data = await fetchJson("/api/mcp/test", { method: "POST", body: payload });
      setStatus(`${text("测试通过", "Test ok")}: ${summarizeMcpTest(data)}`);
    } catch (error) {
      setStatus(`${text("测试失败", "Test failed")}: ${String(error.message || error)}`);
    }
  }

  async function testSavedServer(name) {
    try {
      const data = await fetchJson(`/api/mcp/servers/${encodeURIComponent(name)}/test`, { method: "POST" });
      setStatus(`${text("测试通过", "Test ok")}: ${summarizeMcpTest(data)}`);
    } catch (error) {
      setStatus(`${text("测试失败", "Test failed")}: ${String(error.message || error)}`);
    }
  }

  function buildMcpPayload(formValue) {
    const base = {
      name: formValue.name.trim(),
      enabled: formValue.enabled,
      transport: formValue.transport,
      timeout: Number(formValue.timeout),
      sse_read_timeout: Number(formValue.sse_read_timeout),
    };
    if (formValue.transport === "stdio") {
      return {
        ...base,
        command: formValue.command.trim(),
        args: formValue.argsText.split("\n").map((item) => item.trim()).filter(Boolean),
        env: headersObject(formValue.envRows),
      };
    }
    return {
      ...base,
      url: cleanInputUrl(formValue.url),
      headers: headersObject(formValue.headerRows),
    };
  }

  async function deleteServer(name) {
    try {
      await fetchJson(`/api/mcp/servers/${encodeURIComponent(name)}`, { method: "DELETE" });
      setStatus(`${text("已删除", "Deleted")} ${name}`);
      await refresh();
    } catch (error) {
      setStatus(`${text("删除失败", "Delete failed")}: ${String(error.message || error)}`);
    }
  }

  const servers = useMemo(() => Object.entries(config.servers || {}), [config]);
  return (
    <section className="mcpLayout" style={{ gridTemplateColumns: splitColumns(null, "r", formPane) }}>
      <div className="serverListPane">
        <div className="editorHeader"><h2>{text("MCP 服务器", "MCP SERVERS")}</h2><button className="secondaryButton" onClick={refresh} type="button"><RefreshCw size={16} /><span>{text("刷新", "Refresh")}</span></button></div>
        {servers.map(([name, server]) => (
          <article className="serverItem" key={name}>
            <div><strong>{name}</strong><span>{server.enabled ? text("已启用", "enabled") : text("已禁用", "disabled")}</span></div>
            <code>{server.transport || "stdio"} {server.url || server.command || ""}</code>
            <div className="rowActions">
              <button className="secondaryButton" onClick={() => editServer(name, server)} type="button"><FileText size={15} /><span>{text("编辑", "Edit")}</span></button>
              <button className="secondaryButton" onClick={() => testSavedServer(name)} type="button"><Plug size={15} /><span>{text("测试", "Test")}</span></button>
              <button className="dangerButton" onClick={() => deleteServer(name)} type="button"><Trash2 size={15} /><span>{text("删除", "Delete")}</span></button>
            </div>
          </article>
        ))}
      </div>
      <Resizer pane={formPane} label={text("调整表单宽度", "Resize form")} />
      <form className="mcpForm" onSubmit={saveServer} aria-hidden={formPane.collapsed || undefined}>
        <h2>{text("新增或编辑服务器", "ADD OR EDIT SERVER")}</h2>
        <label><span>{text("名称", "Name")}</span><input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="rent" /></label>
        <SelectField label={text("传输方式", "Transport")} value={form.transport} onChange={(value) => setForm({ ...form, transport: value })} values={choices("transport", text)} />
        {form.transport === "stdio" ? (
          <>
            <label><span>{text("命令", "Command")}</span><input value={form.command} onChange={(event) => setForm({ ...form, command: event.target.value })} placeholder="node" /></label>
            <label><span>{text("参数，每行一个", "Args, one per line")}</span><textarea value={form.argsText} onChange={(event) => setForm({ ...form, argsText: event.target.value })} placeholder={"server.js\n--port\n5050"} /></label>
            <KeyValueEditor title="Env" label={text("环境变量", "Env")} rows={form.envRows} setForm={setForm} field="envRows" text={text} />
          </>
        ) : (
          <>
            <label><span>URL</span><input value={form.url} onChange={(event) => setForm({ ...form, url: event.target.value })} placeholder="http://127.0.0.1:5050/mcp" /></label>
            <KeyValueEditor title="Headers" label={text("请求头", "Headers")} rows={form.headerRows} setForm={setForm} field="headerRows" text={text} />
          </>
        )}
        <label><span>{text("超时", "Timeout")}</span><input type="number" min="1" value={form.timeout} onChange={(event) => setForm({ ...form, timeout: event.target.value })} /></label>
        <label><span>{text("SSE 读取超时", "SSE Read Timeout")}</span><input type="number" min="1" value={form.sse_read_timeout} onChange={(event) => setForm({ ...form, sse_read_timeout: event.target.value })} /></label>
        <label className="checkLine"><input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} /><span>{text("启用", "Enabled")}</span></label>
        <div className="formActions">
          <button className="primaryButton" type="submit"><Save size={16} /><span>{text("保存服务器", "Save Server")}</span></button>
          <button className="secondaryButton" type="button" onClick={testCurrentServer}><Plug size={16} /><span>{text("测试连接", "Test Connection")}</span></button>
        </div>
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </form>
    </section>
  );
}

function KeyValueEditor({ title, label = title, rows, setForm, field, text = (zh) => zh }) {
  return (
    <div className="headerEditor">
      <div className="labelText">{label}</div>
      {rows.map((row, index) => (
        <div className="headerRow" key={index}>
          <input value={row.key} onChange={(event) => updateHeaderRow(index, "key", event.target.value, setForm, field)} placeholder={text("键", "Key")} />
          <input value={row.value} onChange={(event) => updateHeaderRow(index, "value", event.target.value, setForm, field)} placeholder={text("值", "Value")} />
          <button className="iconButton danger" type="button" onClick={() => removeHeaderRow(index, setForm, field)} title={`${text("移除", "Remove")} ${label}`}>
            <Trash2 size={15} />
          </button>
        </div>
      ))}
      <button className="secondaryButton" type="button" onClick={() => addHeaderRow(setForm, field)}><Plus size={16} /><span>{text("新增", "Add")} {label}</span></button>
    </div>
  );
}

async function readSse(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(6)));
    }
  }
}

// The text of a FastAPI error ("HTTP 400: {"detail": "..."}") without the wrapping.
function errorDetail(error) {
  const raw = String(error?.message || error);
  const start = raw.indexOf("{");
  if (start >= 0) {
    try {
      const detail = JSON.parse(raw.slice(start)).detail;
      if (detail) return typeof detail === "string" ? detail : JSON.stringify(detail);
    } catch {
      /* not JSON */
    }
  }
  return raw;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}

function normalizeOptions(options) {
  return { ...options, web_search_mode: normalizeWebMode(options.web_search_mode), conversation_mode: normalizeConversationMode(options.conversation_mode), model: options.model || null, system_prompt: options.system_prompt || null };
}

function normalizeConversationMode(mode) {
  return mode === "chat" ? "ask" : (mode || "agent");
}

function modelValue(item) {
  if (typeof item === "string") return item;
  return item?.id || item?.alias || "";
}

function stringifyEventText(event, fallback) {
  if (["tool_call", "assistant_progress", "question_required", "approval_required", "ai_review", "plan_updated", "plan_ready", "plan_decision", "approval_request", "parameters_changed", "error", "stopped", "user", "assistant"].includes(event.type)) {
    return fallback;
  }
  return JSON.stringify(event, null, 2);
}

function formatQuestionResponseForUser(response, text) {
  if (response.status === "declined") return text("我选择不回答这个问题。", "I declined to answer this question.");
  if (response.status === "redirected") return `${text("改变方向：", "Change direction: ")}${response.direction}`;
  const parts = [];
  if (response.selected.length > 0) parts.push(response.selected.join(text("、", ", ")));
  if (response.text) parts.push(response.text);
  return parts.join("\n") || text("已回答。", "Answered.");
}

function eventLabel(type) {
  const labels = {
    user: "USER",
    assistant: "ASSISTANT",
    assistant_progress: "THINKING",
    tool_call: "TOOL",
    question_required: "QUESTION",
    approval_required: "APPROVAL",
    approval_request: "APPROVAL",
    ai_review: "AI REVIEW",
    error: "ERROR",
    stopped: "STOPPED",
    python: "PYTHON",
    rag: "RAG",
    mcp: "MCP",
    curl: "HTTP",
    webSearch: "WEB",
    fileEditor: "FILE",
    fileReader: "READER",
    settings_changed: "SETTINGS",
    parameters_changed: "PARAMETERS",
    parameterSave: "PARAMETERS",
    plan_updated: "PLAN DRAFT",
    plan_ready: "PLAN READY",
    plan_decision: "PLAN REVIEW",
  };
  return labels[type] || String(type).toUpperCase();
}

function extractImages(text) {
  const patterns = [
    /!\[[^\]]*]\(([^)]+)\)/g,
    /image:\s*(https?:\/\/[^\s<>"')]+)/gi,
    /(https?:\/\/[^\s<>"')]+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?[^\s<>"')]+)?)/gi,
    /([A-Za-z]:[\\/][^\n\r"'<>|]+\bpython_runs[\\/][^\n\r"'<>|]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /([A-Za-z]:[\\/][^\n\r"'<>|]+\bmcp_artifacts[\\/][^\n\r"'<>|]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\/runtime\/python_runs\/[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\\runtime\\python_runs\\[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\/runtime\/mcp_artifacts\/[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\\runtime\\mcp_artifacts\\[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\/runtime\/uploads\/[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\\runtime\\uploads\\[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(\/api\/uploads\/[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?[^\s<>"')]+)?)/gi,
  ];
  const found = [];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.push(cleanImageSrc(match[1]));
  }
  return Array.from(new Set(found.filter(Boolean)));
}

function extractLooseImages(text) {
  const markdownImages = new Set();
  for (const match of text.matchAll(/!\[[^\]]*]\(([^)]+)\)/g)) {
    markdownImages.add(cleanImageSrc(match[1]));
  }
  return extractImages(text).filter((src) => !markdownImages.has(src));
}

function extractReferences(text) {
  const refs = [];
  for (const match of text.matchAll(/https?:\/\/[^\s<>"')]+/gi)) refs.push(cleanImageSrc(match[0]));
  for (const match of text.matchAll(/(?:data|backend)\/[^\s)]+\.(?:md|txt|json|yaml|yml|html|png|jpg|jpeg|gif|webp|svg)/gi)) {
    refs.push(cleanImageSrc(match[0]));
  }
  return Array.from(new Set(refs)).slice(0, 8);
}

function normalizeReferenceHref(ref) {
  if (/^https?:\/\//i.test(ref)) return ref;
  if (/\.(?:html|png|jpg|jpeg|gif|webp|svg)$/i.test(ref)) return normalizeImageSrc(ref);
  return "#";
}

function cleanImageSrc(src) {
  return String(src || "").trim().replace(/[.,;:]+$/g, "");
}

function normalizeImageSrc(src) {
  if (!src) return "";
  if (/^https?:\/\//i.test(src) || src.startsWith("data:")) return src;
  if (src.startsWith("/api/")) return `${API_BASE}${src}`;
  return `${API_BASE}/api/artifact?path=${encodeURIComponent(src.replaceAll("\\", "/"))}`;
}

function imageDownloadName(src) {
  const clean = cleanImageSrc(src).split("?", 1)[0].replaceAll("\\", "/");
  const rawName = clean.split("/").filter(Boolean).pop() || "image";
  let name = rawName;
  try {
    name = decodeURIComponent(rawName);
  } catch {
    name = rawName;
  }
  return name || "image";
}

function toolMentionOptions(text) {
  return [
    { label: text("工具 Python", "Tool Python"), token: "@tool:python" },
    { label: text("工具 文件阅读", "Tool File Reader"), token: "@tool:fileReader" },
    { label: text("工具 RAG", "Tool RAG"), token: "@tool:rag" },
    { label: text("工具 Web", "Tool Web"), token: "@tool:webSearch" },
    { label: text("工具 HTTP", "Tool HTTP"), token: "@tool:curl" },
    { label: text("工具 文件编辑", "Tool File Editor"), token: "@tool:fileEditor" },
    { label: text("工具 MCP", "Tool MCP"), token: "@tool:mcp" },
    { label: text("工具 变量存储", "Tool Saved Parameters"), token: "@tool:parameterSave" },
    { label: text("工具 历史", "Tool History"), token: "@tool:history" },
    { label: text("工具设置", "Tool Settings"), token: "@tool:settings" },
  ];
}

function conversationMentionOptions(conversations, text) {
  return conversations.map((item) => ({
    label: `${text("历史", "History")} ${item.title || item.id}`,
    token: `@history:${item.id}`,
  }));
}

function fileMentionOptions(kind, files, text) {
  const labelMap = {
    memory: text("记忆", "Memory"),
    skill: text("技能", "Skill"),
    knowledge: text("知识", "Knowledge"),
    instruction: text("指令", "Instruction"),
  };
  return files.map((item) => {
    const path = typeof item === "string" ? item : item.path;
    const scope = typeof item === "string" ? "" : item.scope;
    return {
      label: `${labelMap[kind] || kind} ${scope ? `${scope} ` : ""}${path}`,
      token: `@file:${path}`,
    };
  });
}

function mergeMentionOptions(left, right) {
  const map = new Map(left.map((item) => [item.token, item]));
  for (const item of right) map.set(item.token, item);
  return Array.from(map.values());
}

function cleanInputUrl(value) {
  let text = String(value || "").trim().replaceAll("\\_", "_");
  if (text.startsWith("[") && text.includes("](") && text.endsWith(")")) {
    text = text.split("](", 2)[1].slice(0, -1).trim();
  }
  return text;
}

function automationPayloadFromForm(form) {
  return {
    title: form.title,
    action: form.action,
    enabled: form.enabled,
    prompt: form.prompt,
    code: form.code,
    mcp_server: form.mcp_server,
    mcp_tool: form.mcp_tool,
    schedule: parseJsonObject(form.scheduleText, "schedule"),
    mcp_arguments: parseJsonObject(form.mcpArgumentsText, "mcp_arguments"),
    mcp_config: parseJsonObject(form.mcpConfigText, "mcp_config"),
  };
}

function scheduleSummary(item, text) {
  const kind = item.schedule_kind || item.schedule?.kind || text("未知", "unknown");
  const nextRun = item.next_run_at || item.schedule?.nextRunAt || text("未设置", "not set");
  return `${text("时间", "Schedule")}: ${kind} · ${text("下次", "Next")}: ${nextRun}`;
}

function safeScheduleValue(rawText, key) {
  try {
    const schedule = parseJsonObject(rawText, "schedule");
    return schedule[key] ? String(schedule[key]) : "";
  } catch {
    return "";
  }
}

function formatRunRecord(run) {
  const payload = { ...run };
  delete payload.path;
  return JSON.stringify(payload, null, 2);
}

function parseJsonObject(rawText, fieldName) {
  const value = rawText.trim() ? JSON.parse(rawText) : {};
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${fieldName} must be an object`);
  }
  return value;
}

function headerRowsFromObject(headers) {
  const rows = Object.entries(headers || {}).map(([key, value]) => ({ key, value: String(value) }));
  return rows.length > 0 ? rows : [{ key: "Authorization", value: "Bearer " }];
}

function headersObject(rows) {
  const headers = {};
  for (const row of rows) {
    const key = String(row.key || "").trim();
    if (key) headers[key] = String(row.value || "");
  }
  return headers;
}

function summarizeMcpTest(data) {
  const text = data?.result?.response || "";
  try {
    const parsed = JSON.parse(text);
    const tools = parsed?.result?.tools;
    if (Array.isArray(tools)) return `${tools.length} tools`;
  } catch {
    // Keep the raw prefix below for non-JSON MCP responses.
  }
  return String(text || "connected").slice(0, 180);
}

function updateHeaderRow(index, field, value, setForm, rowsField = "headerRows") {
  setForm((current) => {
    const nextRows = current[rowsField].map((row, rowIndex) => (
      rowIndex === index ? { ...row, [field]: value } : row
    ));
    return { ...current, [rowsField]: nextRows };
  });
}

function addHeaderRow(setForm, rowsField = "headerRows") {
  setForm((current) => ({
    ...current,
    [rowsField]: [...current[rowsField], { key: "", value: "" }],
  }));
}

function removeHeaderRow(index, setForm, rowsField = "headerRows") {
  setForm((current) => {
    const nextRows = current[rowsField].filter((_, rowIndex) => rowIndex !== index);
    return { ...current, [rowsField]: nextRows.length > 0 ? nextRows : [{ key: "", value: "" }] };
  });
}

function findLastIndex(items, predicate) {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (predicate(items[index], index)) return index;
  }
  return -1;
}

function deepMerge(base, patch) {
  const result = { ...(base || {}) };
  for (const [key, value] of Object.entries(patch || {})) {
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      result[key] &&
      typeof result[key] === "object" &&
      !Array.isArray(result[key])
    ) {
      result[key] = deepMerge(result[key], value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

function useLabel(language) {
  return (key) => {
    const item = TEXT[key] || [key, key];
    if (language === "en") return item[1];
    if (language === "both") return `${item[0]} / ${item[1]}`;
    return item[0];
  };
}

function useText(language) {
  return (zh, en) => {
    if (language === "en") return en;
    if (language === "both") return `${zh} / ${en}`;
    return zh;
  };
}


export default App;
