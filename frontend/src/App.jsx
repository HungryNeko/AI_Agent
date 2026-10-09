import React, { useEffect, useMemo, useRef, useState } from "react";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  Archive,
  AtSign,
  Check,
  ChevronRight,
  Code2,
  Database,
  Download,
  FileText,
  GitBranch,
  Image,
  Key,
  Languages,
  Maximize2,
  MessageSquare,
  Minus,
  Paperclip,
  Pause,
  Play,
  Plug,
  Plus,
  RefreshCw,
  Save,
  Send,
  Settings,
  Square,
  Sun,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import "katex/dist/katex.min.css";
import "./styles.css";
import { currentOperation, toolPreviewText } from "./operationPreview.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";

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
  autoApproval: ["自动批准", "Auto approval"],
  system: ["系统", "System"],
};

function App() {
  const [tab, setTab] = useState("chat");
  const [models, setModels] = useState([]);
  const [options, setOptionsState] = useState(emptyOptions);
  const [theme, setThemeState] = useState("system");
  const [language, setLanguageState] = useState("zh");
  const pendingSettingsPatchRef = useRef({});
  const settingsSaveTimerRef = useRef(null);
  const label = useLabel(language);
  const text = useText(language);

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

      {tab === "chat" && (
        <ChatView
          models={models}
          options={options}
          setOptions={setOptions}
          label={label}
          text={text}
          onSettingsChanged={reloadSettings}
          onModelsChanged={reloadModels}
        />
      )}
      {tab === "data" && <DataView text={text} models={models} />}
      {tab === "config" && (
        <ConfigView text={text} onSaved={reloadModels} />
      )}
      {tab === "automation" && <AutomationView options={options} setOptions={setOptions} text={text} />}
      {tab === "mcp" && <McpView text={text} />}
      {tab === "tools" && <CustomToolsView text={text} />}
      {tab === "system" && <SystemView theme={theme} setTheme={setTheme} language={language} setLanguage={setLanguage} text={text} />}
    </main>
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

function ChatView({ models, options, setOptions, label, text, onSettingsChanged, onModelsChanged }) {
  const [message, setMessage] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [events, setEvents] = useState([]);
  const [state, setState] = useState(null);
  const [conversationId, setConversationId] = useState("");
  const [conversationModel, setConversationModel] = useState("");
  const [conversations, setConversations] = useState([]);
  const [historyStatus, setHistoryStatus] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mentionOptions, setMentionOptions] = useState([]);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionOpen, setMentionOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [inputMode, setInputMode] = useState("queue");
  const [queuedMessages, setQueuedMessages] = useState([]);
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
  }, []);

  useEffect(() => {
    if (atBottomRef.current) outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight });
  }, [events]);

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
    const data = await fetchJson(`/api/conversations/${encodeURIComponent(id)}`);
    setConversationId(data.id || id);
    setEvents(data.events || []);
    setState(data.state || null);
    setConversationModel(data.state?.model || "");
    setPendingQuestion(data.state?.question_pending || null);
    setAttachments([]);
    setHistoryStatus("");
  }

  async function renameConversation(item) {
    const title = window.prompt("重命名对话", item.title || "");
    if (!title) return;
    await fetchJson(`/api/conversations/${encodeURIComponent(item.id)}`, { method: "PATCH", body: { title } });
    await refreshConversations();
  }

  async function deleteConversation(id) {
    if (!window.confirm("删除这条对话历史？")) return;
    await fetchJson(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (conversationId === id) newConversation();
    await refreshConversations();
  }

  function newConversation() {
    if (busy) return;
    setConversationId("");
    setEvents([]);
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

  async function runChatTurn(nextMessage, nextAttachments = [], displayMessage = nextMessage, overrides = {}) {
    if (!nextMessage.trim() || busy) return;
    atBottomRef.current = true;
    setBusy(true);
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
        setEvents((items) => [...items, eventData]);
        if (["assistant", "stopped"].includes(eventData.type) || (eventData.type === "error" && eventData.terminal)) {
          streamCompleted = true;
        }
        if (eventData.conversation_id) setConversationId(eventData.conversation_id);
        if (eventData.type === "question_required") setPendingQuestion(eventData);
        if (eventData.type === "model_changed") setConversationModel(eventData.text || "");
        if (eventData.type === "models_changed") onModelsChanged?.().catch(() => {});
        if (eventData.type === "assistant" && eventData.state) {
          setState(eventData.state);
          setConversationModel(eventData.state.model || conversationModel || "");
          setPendingQuestion(eventData.state.question_pending || null);
        }
        if (eventData.type === "settings_changed") onSettingsChanged?.().catch(() => {});
      });
      if (!streamCompleted) {
        throw new Error(text("响应流意外结束，后端没有返回最终结果。", "The response stream ended without a final result."));
      }
      await refreshConversations();
    } catch (error) {
      if (!stopRequestedRef.current && error?.name !== "AbortError") {
        setEvents((items) => [...items, { type: "error", terminal: true, text: String(error.message || error) }]);
      }
    } finally {
      if (activeRunIdRef.current === runId) {
        abortRef.current = null;
        activeRunIdRef.current = "";
        stopRequestedRef.current = false;
        setBusy(false);
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
    window.setTimeout(() => refreshConversations().catch(() => {}), 500);
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

  async function branchConversation(eventIndex) {
    if (!conversationId || busy) return;
    const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/branch`, {
      method: "POST",
      body: { event_index: eventIndex },
    });
    setConversationId(data.id);
    setEvents(data.events || []);
    setState(data.state || null);
    setConversationModel(data.state?.model || conversationModel || "");
    setHistoryStatus(text("已创建对话分支", "Conversation branch created"));
    await refreshConversations();
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

  async function reviewPlan(decision) {
    if (!conversationId || busy) return;
    const data = await fetchJson(`/api/conversations/${encodeURIComponent(conversationId)}/plan/decision`, {
      method: "POST",
      body: { decision },
    });
    setState(data.state || null);
    setEvents(data.events || []);
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

  return (
    <section className="chatLayout">
      <aside className="historyPane">
        <div className="paneHeader">
          <div className="sidebarTabs">
            <button className={!settingsOpen ? "sidebarTab active" : "sidebarTab"} onClick={() => setSettingsOpen(false)} type="button">
              {label("history")}
            </button>
            <button className={settingsOpen ? "sidebarTab active" : "sidebarTab"} onClick={() => setSettingsOpen(true)} type="button">
              {label("settings")}
            </button>
          </div>
          {!settingsOpen && (
            <button className="iconButton neutral" onClick={newConversation} type="button" title={label("newChat")}>
              <Plus size={17} />
            </button>
          )}
        </div>
        {settingsOpen ? (
          <SettingsPanel
            models={models}
            options={options}
            setOptions={setOptions}
            currentModel={conversationModel || options.model}
            setCurrentModel={setConversationModel}
            text={text}
            clearState={newConversation}
          />
        ) : (
          <div className="historyList">
            {conversations.map((item) => (
              <div className={conversationId === item.id ? "historyRow active" : "historyRow"} key={item.id}>
                <button className="historyItem" onClick={() => openConversation(item.id)} type="button">
                  <MessageSquare size={15} />
                  <span>{item.title || "未命名"}</span>
                </button>
                <button className="miniButton" onClick={() => renameConversation(item)} type="button" title="重命名">改</button>
                <button className="miniButton dangerMini" onClick={() => deleteConversation(item.id)} type="button" title="删除">删</button>
              </div>
            ))}
          </div>
        )}
        {historyStatus && <p className="miniStatus">{historyStatus}</p>}
      </aside>
      <div className="chatPane">
        <div className="chatHeader">
          <div>
            <h2>{conversationId ? "对话" : label("newChat")}</h2>
            <span>{conversationModel || options.model || "默认模型"}</span>
          </div>
          <button className="secondaryButton" onClick={compressConversation} disabled={!conversationId || busy} type="button">
            <Archive size={16} />
            <span>{label("compress")}</span>
          </button>
        </div>
        <div className="stream" ref={outputRef} onScroll={trackScroll}>
          {events.length === 0 && <div className="emptyState">输入消息，或用 @ 指定工具、历史、技能、记忆和知识。</div>}
          <StreamEvents events={events} busy={busy} onPreviewImage={setPreviewImage} onBranch={branchConversation} text={text} />
          <PlanReviewCard plan={state?.plan} busy={busy} onApprove={() => reviewPlan("approved")} onReject={() => reviewPlan("rejected")} onRevise={revisePlan} onSave={savePlanCopy} text={text} />
        </div>
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
            <div className="composerFooter">
              <div className="composerTools">
                <button className="composerIcon" type="button" onClick={() => fileInputRef.current?.click()} title={label("uploadFile")}><Paperclip size={17} /></button>
                <button className="composerIcon" type="button" onClick={() => imageInputRef.current?.click()} title={label("uploadImage")}><Image size={17} /></button>
                <select className="composerPicker modePicker" value={normalizeConversationMode(options.conversation_mode)} onChange={(event) => setOptions((current) => ({ ...current, conversation_mode: event.target.value }))} title={text("对话模式", "Conversation mode")} aria-label={text("对话模式", "Conversation mode")}>
                  <option value="ask">{text("问答", "Ask")}</option>
                  <option value="plan">{text("计划", "Plan")}</option>
                  <option value="agent">Agent</option>
                </select>
                <select className="composerPicker modelPicker" value={conversationModel || options.model || ""} onChange={(event) => setConversationModel(event.target.value)} title={text("当前模型", "Current model")} aria-label={text("当前模型", "Current model")}>
                  <option value="">{text("默认模型", "Default model")}</option>
                  {models.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
                </select>
                <button className={options.file_editor_approval === "auto" ? "composerIcon active" : "composerIcon"} type="button" onClick={() => setOptions((current) => ({ ...current, file_editor_approval: current.file_editor_approval === "auto" ? "manual" : "auto" }))} title={label("autoApproval")} aria-pressed={options.file_editor_approval === "auto"}><Check size={17} /></button>
              </div>
              <div className="composerActions">
                <select className="composerPicker queuePicker" value={inputMode} onChange={(event) => setInputMode(event.target.value)} title={text("运行中发送方式", "Send while running")} aria-label={text("运行中发送方式", "Send while running")}>
                  <option value="queue">{text("排队", "Queue")}</option>
                  <option value="insert">{text("打断", "Interrupt")}</option>
                </select>
                {busy && <button className="composerIcon" type="button" onClick={togglePause} title={paused ? text("继续", "Resume") : text("暂停", "Pause")}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>}
                {busy && <button className="composerIcon dangerIcon" type="button" onClick={stopOutput} title={label("stop")}><Square size={15} /></button>}
                <button className="composerSend" type="submit" disabled={!message.trim() && attachments.length === 0} title={busy ? (inputMode === "insert" ? text("打断并发送", "Interrupt and send") : text("加入队列", "Add to queue")) : label("send")} aria-label={label("send")}><Send size={17} /></button>
              </div>
            </div>
          </div>
          <input ref={fileInputRef} type="file" className="hiddenInput" onChange={uploadSelectedFile} multiple />
          <input ref={imageInputRef} type="file" accept="image/*" className="hiddenInput" onChange={uploadSelectedFile} multiple />
        </form>
        <QuestionDialog
          key={`${pendingQuestion?.run_id || "saved"}:${pendingQuestion?.question || ""}`}
          question={pendingQuestion}
          busy={busy}
          onRespond={respondToQuestion}
          text={text}
        />
        <ImagePreview image={previewImage} onClose={() => setPreviewImage(null)} />
      </div>
    </section>
  );
}

function SettingsPanel({ models, options, setOptions, currentModel, setCurrentModel, clearState, text }) {
  const update = (key, value) => setOptions((current) => ({ ...current, [key]: value }));
  return (
    <div className="settingsPane embeddedSettings">
      <div className="paneHeader">
        <h2>{text("设置", "SETTINGS")}</h2>
        <Settings size={17} />
      </div>
      <label>
        <span>{text("模型", "Model")}</span>
        <select value={currentModel || ""} onChange={(event) => setCurrentModel(event.target.value)}>
          <option value="">{text("默认", "Default")}</option>
          {models.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
        </select>
      </label>
      <QuestionLevelControl value={options.question_mode} onChange={(value) => update("question_mode", value)} text={text} />
      <button
        className={options.developer_mode ? "developerToggle active" : "developerToggle"}
        type="button"
        onClick={() => update("developer_mode", !options.developer_mode)}
        aria-pressed={options.developer_mode}
      >
        <Code2 size={16} />
        <span>{text("开发者模式", "Developer mode")}</span>
        <strong>{options.developer_mode ? "ON" : "OFF"}</strong>
      </button>
      <label>
        <span>{text("额外提示", "Extra Prompt")}</span>
        <textarea className="smallTextArea" value={options.system_prompt} onChange={(event) => update("system_prompt", event.target.value)} placeholder={text("仅当前对话追加到首个 system prompt", "Append only to the first system prompt in this chat")} />
      </label>
      <div className="settingGrid">
        <SelectField label={text("联网", "Web")} value={options.web_search_mode} onChange={(value) => update("web_search_mode", value)} values={["off", "auto"]} />
        <SelectField label={text("搜索", "Search")} value={options.web_search_provider} onChange={(value) => update("web_search_provider", value)} values={["duckduckgo", "searxng", "tavily"]} />
        <SelectField label={text("自动切换", "Search Fallback")} value={String(options.web_search_auto_switch)} onChange={(value) => update("web_search_auto_switch", value === "true")} values={["true", "false"]} />
        <SelectField label="RAG" value={options.rag_mode} onChange={(value) => update("rag_mode", value)} values={["off", "on", "auto"]} />
        <SelectField label="HTTP" value={options.curl_mode} onChange={(value) => update("curl_mode", value)} values={["off", "auto"]} />
        <SelectField label="Python" value={options.python_mode} onChange={(value) => update("python_mode", value)} values={["off", "auto"]} />
        <SelectField label={text("文件阅读", "File Reader")} value={options.file_reader_mode} onChange={(value) => update("file_reader_mode", value)} values={["off", "auto"]} />
        <SelectField label={text("文件", "File")} value={options.file_editor_mode} onChange={(value) => update("file_editor_mode", value)} values={["off", "auto"]} />
        <SelectField label={text("批准", "Approval")} value={options.file_editor_approval} onChange={(value) => update("file_editor_approval", value)} values={["manual", "auto", "aiReview", "readOnly"]} />
        <SelectField label="MCP" value={options.mcp_mode} onChange={(value) => update("mcp_mode", value)} values={["off", "auto"]} />
        <SelectField label={text("历史", "History")} value={options.history_mode} onChange={(value) => update("history_mode", value)} values={["off", "auto"]} />
        <SelectField label={text("自动化", "Automation")} value={options.automation_mode} onChange={(value) => update("automation_mode", value)} values={["off", "auto"]} />
        <label>
          <span>{text("对话模式", "Mode")}</span>
          <select value={normalizeConversationMode(options.conversation_mode)} onChange={(event) => update("conversation_mode", event.target.value)}>
            <option value="ask">{text("问答 - 只读理解与查询", "Ask - read-only answers")}</option>
            <option value="plan">{text("计划 - 研究并等待批准", "Plan - research and review")}</option>
            <option value="agent">{text("Agent - 执行、测试、迭代", "Agent - implement and verify")}</option>
          </select>
        </label>
      </div>
      <div className="checkGrid">
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_memory} onChange={(event) => update("rag_include_memory", event.target.checked)} /><span>{text("RAG 记忆", "RAG Memory")}</span></label>
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_skills} onChange={(event) => update("rag_include_skills", event.target.checked)} /><span>{text("RAG 技能", "RAG Skill")}</span></label>
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_knowledge} onChange={(event) => update("rag_include_knowledge", event.target.checked)} /><span>{text("RAG 知识", "RAG Knowledge")}</span></label>
      </div>
      <label>
        <span>{text("工具轮次", "Tool Rounds")}</span>
        <input type="number" min="-1" value={options.max_tool_rounds} onChange={(event) => update("max_tool_rounds", Number(event.target.value))} />
        <small>{text("-1 = 无限制，0 = 禁用工具", "-1 = unlimited, 0 = tools disabled")}</small>
      </label>
      <button className="secondaryButton" type="button" onClick={clearState} title={text("重置对话", "Clear conversation")}>
        <RefreshCw size={16} />
        <span>{text("重置对话", "Reset")}</span>
      </button>
    </div>
  );
}

function SelectField({ label, value, onChange, values, icon = null }) {
  return (
    <label>
      <span>{icon}{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {values.map((item) => <option key={item} value={item}>{item}</option>)}
      </select>
    </label>
  );
}

function StreamEvents({ events, busy = false, onPreviewImage, onBranch, text }) {
  const turns = [];
  let current = [];
  for (const [eventIndex, rawEvent] of events.entries()) {
    const event = { ...rawEvent, eventIndex };
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
      text={text}
    />
  ));
}

function PlanReviewCard({ plan, busy, onApprove, onReject, onRevise, onSave, text }) {
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
        {ready && <button className="secondaryButton dangerButton" type="button" onClick={onReject} disabled={busy}>{text("拒绝", "Reject")}</button>}
        {ready && <button className="primaryButton" type="button" onClick={onApprove} disabled={busy}><Check size={16} /><span>{text("批准并执行", "Approve and implement")}</span></button>}
      </div>
    </section>
  );
}

function QuestionLevelControl({ value, onChange, text }) {
  const levels = ["off", "light", "heavy"];
  const index = Math.max(0, levels.indexOf(value));
  return (
    <label className="questionLevel">
      <span>{text("提问强度", "Clarification")}</span>
      <input
        type="range"
        min="0"
        max="2"
        step="1"
        value={index}
        onChange={(event) => onChange(levels[Number(event.target.value)])}
      />
      <div className="questionLevelLabels" aria-hidden="true">
        <span>{text("跳过", "Skip")}</span>
        <span>{text("轻度", "Light")}</span>
        <span>{text("重度", "Heavy")}</span>
      </div>
    </label>
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

function StreamTurn({ events, running = false, onPreviewImage, onBranch, text }) {
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
      {userEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} onPreviewImage={onPreviewImage} />)}
      {(operationEvents.length > 0 || active) && (
        <section className="operationGroup">
          <details className="operationDetails" open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
            <summary>
              <ChevronRight size={14} className="operationChevron" />
              {running && !completed && <span className="runningDot" />}
              <span>{completed ? text("执行过程", "Activity") : currentWork.attention?.type === "question_required" ? text("等待回答", "Waiting for answer") : currentWork.attention?.type === "approval_required" ? text("等待批准", "Waiting for approval") : text("进行中", "Working")}</span>
              <small>{operationEvents.length} {text("条记录", "events")}</small>
            </summary>
            {operationEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} compact onPreviewImage={onPreviewImage} />)}
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
      {finalEvent && String(finalEvent.text || "").trim() && <StreamEvent event={finalEvent} onPreviewImage={onPreviewImage} onBranch={onBranch} />}
    </div>
  );
}

function StreamEvent({ event, compact = false, onPreviewImage, onBranch }) {
  const type = event.type || "event";
  const fallback = event.text || event.query || event.url || event.question || "";
  const body = stringifyEventText(event, fallback);
  const images = extractLooseImages(body);
  return (
    <article className={`event event-${type}${compact ? " compact" : ""}`}>
      <div className="eventType">{eventLabel(type)}{type === "assistant" && onBranch && <button className="branchButton" type="button" onClick={() => onBranch(event.eventIndex)} title="Branch conversation"><GitBranch size={13} /></button>}</div>
      <MarkdownText text={body} onPreviewImage={onPreviewImage} />
      {event.attachments?.length > 0 && <AttachmentList attachments={event.attachments} onPreviewImage={onPreviewImage} />}
      {images.length > 0 && <ImageStrip images={images} onPreviewImage={onPreviewImage} />}
      {type === "assistant" && <ReferenceList refs={extractReferences(body)} />}
    </article>
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
  const knowledgeUploadRef = useRef(null);

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
  ];
  const isMarkdown = selected.toLowerCase().endsWith(".md");

  return (
    <section className="dataLayout">
      <aside className="fileListPane">
        <div className="segmented vertical">
          {resourceTabs.map(([value, tabLabel]) => (
            <button key={value} className={kind === value ? "active" : ""} onClick={() => setKind(value)} type="button">{tabLabel}</button>
          ))}
        </div>
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
        <input ref={knowledgeUploadRef} className="hiddenInput" type="file" onChange={ingestKnowledgeFile} />
        <div className="fileList">
          {files.map((file) => (
            <button key={file.path} className={selected === file.path ? "fileItem active" : "fileItem"} onClick={() => openFile(file)} type="button">
              <FileText size={15} />
              <span>{file.path}</span>
              <small>{file.scope === "system" ? "SYSTEM" : "USER"}</small>
            </button>
          ))}
        </div>
      </aside>
      <div className="editorPane">
        <div className="editorHeader">
          <div>
            <h2>{selected || "资源文件"}</h2>
            <span>{text(
              `${selectedMeta.writable ? "可编辑用户文件" : "系统级文件只读"}。指令是短规则，记忆是事实，技能是流程，知识是参考文档。`,
              `${selectedMeta.writable ? "Editable user file" : "System file is read-only"}. Instructions are short rules, memory is facts, skills are workflows, and knowledge is reference material.`
            )}</span>
          </div>
          <div className="rowActions">
            <button className="secondaryButton" onClick={() => setPreview((value) => !value)} disabled={!isMarkdown} type="button"><FileText size={16} /><span>{preview ? text("编辑", "Edit") : text("显示 MD", "Show MD")}</span></button>
            <button className="secondaryButton" onClick={renameSelected} disabled={!selectedMeta.writable || !isMarkdown} type="button"><AtSign size={16} /><span>{text("重命名", "Rename")}</span></button>
            <button className="iconButton danger" onClick={deleteSelected} disabled={!selected || !selectedMeta.writable || kind === "instruction"} type="button" title={text("删除", "Delete")}><Trash2 size={16} /></button>
            <button className="primaryButton" onClick={saveFile} disabled={!selected || !selectedMeta.writable} type="button"><Save size={16} /><span>{text("保存", "Save")}</span></button>
          </div>
        </div>
        {preview ? (
          <div className="markdownPreview"><MarkdownText text={content} /></div>
        ) : (
          <textarea className="codeEditor" value={content} onChange={(event) => setContent(event.target.value)} readOnly={!selectedMeta.writable} placeholder={text("选择并编辑指令、记忆、技能或知识文件。", "Choose and edit an instruction, memory, skill, or knowledge file.")} />
        )}
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
      <form className="importPane" onSubmit={importResource}>
        <h2>{text("导入", "IMPORT")}</h2>
        <label><span>{text("名称", "Name")}</span><input value={importName} onChange={(event) => setImportName(event.target.value)} placeholder={kind === "skills" ? "my-skill" : "notes.md"} /></label>
        <label><span>{text("内容", "Content")}</span><textarea value={importContent} onChange={(event) => setImportContent(event.target.value)} /></label>
        <button className="primaryButton" type="submit"><Plus size={16} /><span>{text("导入到当前分类", "Import to current category")}</span></button>
      </form>
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

function ConfigView({ onSaved, text }) {
  const [config, setConfig] = useState({ providers: {} });
  const [selectedProvider, setSelectedProvider] = useState("");
  const [status, setStatus] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [newModel, setNewModel] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    refreshConfig();
  }, []);

  async function refreshConfig(preferredProvider = "") {
    const data = await fetchJson("/api/config");
    const nextConfig = data.config || { providers: {} };
    setConfig(nextConfig);
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
      const data = await fetchJson("/api/config", { method: "PUT", body: { config: payload } });
      setConfig(data.config || config);
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
  }

  const providers = config.providers || {};
  const provider = selectedProvider ? providers[selectedProvider] || {} : {};
  const providerModels = Array.isArray(provider.models) ? provider.models : [];

  return (
    <section className="configLayout">
      <aside className="providerPane">
        <div className="editorHeader">
          <h2>{text("供应商", "PROVIDER")}</h2>
          <button className="iconButton neutral" onClick={addProvider} type="button" title={text("新增", "Add")}><Plus size={16} /></button>
        </div>
        {Object.keys(providers).map((name) => (
          <button key={name} className={selectedProvider === name ? "providerItem active" : "providerItem"} onClick={() => setSelectedProvider(name)} type="button">
            <span><Key size={15} />{name}</span>
            {config.default_provider === name && <small>{text("默认", "DEFAULT")}</small>}
          </button>
        ))}
      </aside>
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
    </section>
  );
}

function CustomToolsView({ text }) {
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
    <section className="configLayout">
      <aside className="providerPane">
        <div className="editorHeader"><h2>{text("自建工具", "CUSTOM TOOLS")}</h2><button className="iconButton neutral" type="button" onClick={() => { setSelected(""); setForm(emptyTool); setStatus(""); }}><Plus size={16} /></button></div>
        {tools.map((item) => <button className={selected === item.name ? "providerItem active" : "providerItem"} key={item.name} type="button" onClick={() => openTool(item.name)}><span><Wrench size={15} />custom__{item.name}</span></button>)}
      </aside>
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
  return (
    <section className="systemLayout">
      <div className="systemPanel">
        <div className="editorHeader">
          <div>
            <h2>{text("系统设置", "SYSTEM SETTINGS")}</h2>
            <span>{text("这些设置只影响界面，不会混入当前对话提示词。", "These settings only affect the UI and are not injected into chat prompts.")}</span>
          </div>
        </div>
        <div className="modelForm">
          <SelectField label={text("主题", "Theme")} value={theme} onChange={setTheme} values={["system", "light", "dark"]} icon={<Sun size={14} />} />
          <SelectField label={text("语言", "Language")} value={language} onChange={setLanguage} values={["zh", "en", "both"]} icon={<Languages size={14} />} />
        </div>
      </div>
    </section>
  );
}

function AutomationView({ options, setOptions, text }) {
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
    <section className="configLayout">
      <aside className="providerPane">
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
          <SelectField label={text("类型", "Action")} value={form.action} onChange={(value) => setForm({ ...form, action: value })} values={["reminder", "llm", "script", "mcp", "configureMcp"]} />
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
    <section className="mcpLayout">
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
      <form className="mcpForm" onSubmit={saveServer}>
        <h2>{text("新增或编辑服务器", "ADD OR EDIT SERVER")}</h2>
        <label><span>{text("名称", "Name")}</span><input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="rent" /></label>
        <SelectField label={text("传输方式", "Transport")} value={form.transport} onChange={(value) => setForm({ ...form, transport: value })} values={["streamable_http", "stdio"]} />
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
          <button className="iconButton" type="button" onClick={() => removeHeaderRow(index, setForm, field)} title={`${text("移除", "Remove")} ${label}`}>
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
  return { ...options, conversation_mode: normalizeConversationMode(options.conversation_mode), model: options.model || null, system_prompt: options.system_prompt || null };
}

function normalizeConversationMode(mode) {
  return mode === "chat" ? "ask" : (mode || "agent");
}

function modelValue(item) {
  if (typeof item === "string") return item;
  return item?.id || item?.alias || "";
}

function stringifyEventText(event, fallback) {
  if (["tool_call", "assistant_progress", "question_required", "approval_required", "ai_review", "plan_updated", "plan_ready", "plan_decision", "parameters_changed", "error", "stopped", "user", "assistant"].includes(event.type)) {
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
