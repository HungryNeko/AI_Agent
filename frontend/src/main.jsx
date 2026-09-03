import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Archive,
  AtSign,
  Check,
  Database,
  FileText,
  Image,
  Key,
  Languages,
  MessageSquare,
  Paperclip,
  Plug,
  Plus,
  RefreshCw,
  Save,
  Send,
  Settings,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";
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
  file_editor_mode: "auto",
  file_editor_approval: "auto",
  mcp_mode: "auto",
  history_mode: "auto",
  automation_mode: "auto",
  max_tool_rounds: 20,
};

const TEXT = {
  chat: ["对话", "Chat"],
  data: ["数据", "Data"],
  config: ["模型", "Models"],
  mcp: ["MCP", "MCP"],
  settings: ["设置", "Settings"],
  history: ["历史", "History"],
  newChat: ["新对话", "New Chat"],
  compress: ["压缩", "Compress"],
  send: ["发送", "Send"],
  running: ["运行中", "Running"],
  uploadFile: ["上传文件", "Upload file"],
  uploadImage: ["上传图片", "Upload image"],
  autoApproval: ["自动批准", "Auto approval"],
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

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    fetchJson("/api/models")
      .then((data) => {
        setModels(data.models || []);
        setOptionsState((current) => ({ ...current, model: current.model || data.defaultModel || "" }));
      })
      .catch(() => setModels([]));
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
          <TabButton active={tab === "mcp"} onClick={() => setTab("mcp")} icon={<Plug size={16} />} label={label("mcp")} />
          <TabButton active={tab === "system"} onClick={() => setTab("system")} icon={<Sun size={16} />} label="系统" />
        </nav>
      </header>

      {tab === "chat" && (
        <ChatView
          models={models}
          options={options}
          setOptions={setOptions}
          label={label}
          onSettingsChanged={reloadSettings}
        />
      )}
      {tab === "data" && <DataView />}
      {tab === "config" && (
        <ConfigView onSaved={() => fetchJson("/api/models").then((data) => setModels(data.models || [])).catch(() => {})} />
      )}
      {tab === "mcp" && <McpView />}
      {tab === "system" && <SystemView theme={theme} setTheme={setTheme} language={language} setLanguage={setLanguage} />}
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

function ChatView({ models, options, setOptions, label, onSettingsChanged }) {
  const [message, setMessage] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [events, setEvents] = useState([]);
  const [state, setState] = useState(null);
  const [conversationId, setConversationId] = useState("");
  const [conversations, setConversations] = useState([]);
  const [historyStatus, setHistoryStatus] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mentionOptions, setMentionOptions] = useState([]);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionOpen, setMentionOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const outputRef = useRef(null);
  const composerRef = useRef(null);
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const atBottomRef = useRef(true);

  useEffect(() => {
    refreshConversations();
  }, []);

  useEffect(() => {
    if (atBottomRef.current) outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight });
  }, [events]);

  function trackScroll() {
    const node = outputRef.current;
    if (!node) return;
    atBottomRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
  }

  async function refreshConversations() {
    try {
      const data = await fetchJson("/api/conversations");
      setConversations(data.conversations || []);
      setMentionOptions((current) => mergeMentionOptions(current, conversationMentionOptions(data.conversations || [])));
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
    setAttachments([]);
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
    if ((!message.trim() && attachments.length === 0) || busy) return;
    atBottomRef.current = true;
    setBusy(true);
    const nextMessage = message.trim() || "请读取这些附件。";
    const nextAttachments = attachments;
    setEvents((items) => [...items, { type: "user", text: nextMessage, attachments: nextAttachments }]);
    setMessage("");
    setAttachments([]);
    setMentionOpen(false);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: nextMessage,
          attachments: nextAttachments,
          conversation_id: conversationId || null,
          state,
          options: normalizeOptions(options),
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      if (!response.body) throw new Error("HTTP response has no stream body");
      await readSse(response.body, (eventData) => {
        setEvents((items) => [...items, eventData]);
        if (eventData.conversation_id) setConversationId(eventData.conversation_id);
        if (eventData.type === "assistant" && eventData.state) setState(eventData.state);
        if (eventData.type === "settings_changed") onSettingsChanged?.().catch(() => {});
      });
      await refreshConversations();
    } catch (error) {
      setEvents((items) => [...items, { type: "error", text: String(error.message || error) }]);
    } finally {
      setBusy(false);
    }
  }

  async function loadMentionOptions() {
    const base = toolMentionOptions();
    try {
      const [history, memory, skills, knowledge] = await Promise.all([
        fetchJson("/api/conversations").catch(() => ({ conversations: [] })),
        fetchJson("/api/data/files?kind=memory").catch(() => ({ items: [], files: [] })),
        fetchJson("/api/data/files?kind=skills").catch(() => ({ items: [], files: [] })),
        fetchJson("/api/data/files?kind=knowledge").catch(() => ({ items: [], files: [] })),
      ]);
      setMentionOptions([
        ...base,
        ...conversationMentionOptions(history.conversations || []),
        ...fileMentionOptions("memory", memory.items || memory.files || []),
        ...fileMentionOptions("skill", skills.items || skills.files || []),
        ...fileMentionOptions("knowledge", knowledge.items || knowledge.files || []),
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
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const response = await fetch(`${API_BASE}/api/uploads?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: file.type ? { "Content-Type": file.type } : undefined,
      body: await file.arrayBuffer(),
    });
    if (!response.ok) {
      setHistoryStatus(`上传失败: ${await response.text()}`);
      return;
    }
    const data = await response.json();
    const item = {
      path: data.path,
      filename: data.filename || file.name,
      content_type: data.content_type || file.type || "application/octet-stream",
    };
    setAttachments((current) => [...current, item]);
    setMessage((current) => `${current}${current ? " " : ""}@file:${item.path}`);
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
            <span>{options.model || "默认模型"}</span>
          </div>
          <button className="secondaryButton" onClick={compressConversation} disabled={!conversationId || busy} type="button">
            <Archive size={16} />
            <span>{label("compress")}</span>
          </button>
        </div>
        <div className="stream" ref={outputRef} onScroll={trackScroll}>
          {events.length === 0 && <div className="emptyState">输入消息，或用 @ 指定工具、历史、技能、记忆和知识。</div>}
          <StreamEvents events={events} />
        </div>
        <form className="composer" onSubmit={sendMessage}>
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
          {attachments.length > 0 && (
            <div className="attachmentTray">
              {attachments.map((item) => (
                <button
                  className="attachmentChip"
                  key={item.path}
                  type="button"
                  onClick={() => setAttachments((current) => current.filter((candidate) => candidate.path !== item.path))}
                  title="移除附件"
                >
                  {item.content_type?.startsWith("image/") ? <Image size={14} /> : <Paperclip size={14} />}
                  <span>{item.filename}</span>
                  <X size={13} />
                </button>
              ))}
            </div>
          )}
          <div className="composerRow">
            <button className="composerIcon" type="button" onClick={() => fileInputRef.current?.click()} title={label("uploadFile")}><Paperclip size={18} /></button>
            <button className="composerIcon" type="button" onClick={() => imageInputRef.current?.click()} title={label("uploadImage")}><Image size={18} /></button>
            <button
              className={options.file_editor_approval === "auto" ? "composerIcon active" : "composerIcon"}
              type="button"
              onClick={() => setOptions((current) => ({ ...current, file_editor_approval: current.file_editor_approval === "auto" ? "manual" : "auto" }))}
              title={label("autoApproval")}
            >
              <Check size={18} />
            </button>
            <textarea
              ref={composerRef}
              value={message}
              onChange={(event) => updateMessage(event.target.value, event.target.selectionStart)}
              onKeyDown={handleComposerKeyDown}
              onFocus={loadMentionOptions}
              placeholder="输入消息，Enter 发送，Shift+Enter 换行..."
            />
            <button className="primaryButton" type="submit" disabled={busy || (!message.trim() && attachments.length === 0)} title="Send">
              {busy ? <span className="spinner" /> : <Send size={18} />}
              <span>{busy ? label("running") : label("send")}</span>
            </button>
          </div>
          <input ref={fileInputRef} type="file" className="hiddenInput" onChange={uploadSelectedFile} />
          <input ref={imageInputRef} type="file" accept="image/*" className="hiddenInput" onChange={uploadSelectedFile} />
        </form>
      </div>
    </section>
  );
}

function SettingsPanel({ models, options, setOptions, clearState }) {
  const update = (key, value) => setOptions((current) => ({ ...current, [key]: value }));
  return (
    <div className="settingsPane embeddedSettings">
      <div className="paneHeader">
        <h2>设置 SETTINGS</h2>
        <Settings size={17} />
      </div>
      <label>
        <span>模型 Model</span>
        <select value={options.model} onChange={(event) => update("model", event.target.value)}>
          <option value="">默认 Default</option>
          {models.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
        </select>
      </label>
      <label>
        <span>额外提示 Extra Prompt</span>
        <textarea className="smallTextArea" value={options.system_prompt} onChange={(event) => update("system_prompt", event.target.value)} placeholder="仅当前对话追加到首个 system prompt" />
      </label>
      <div className="settingGrid">
        <SelectField label="联网 Web" value={options.web_search_mode} onChange={(value) => update("web_search_mode", value)} values={["off", "auto"]} />
        <SelectField label="搜索 Search" value={options.web_search_provider} onChange={(value) => update("web_search_provider", value)} values={["duckduckgo", "searxng", "tavily"]} />
        <SelectField label="自动切换 Search Fallback" value={String(options.web_search_auto_switch)} onChange={(value) => update("web_search_auto_switch", value === "true")} values={["true", "false"]} />
        <SelectField label="RAG" value={options.rag_mode} onChange={(value) => update("rag_mode", value)} values={["off", "on", "auto"]} />
        <SelectField label="HTTP" value={options.curl_mode} onChange={(value) => update("curl_mode", value)} values={["off", "auto"]} />
        <SelectField label="Python" value={options.python_mode} onChange={(value) => update("python_mode", value)} values={["off", "auto"]} />
        <SelectField label="文件 File" value={options.file_editor_mode} onChange={(value) => update("file_editor_mode", value)} values={["off", "auto"]} />
        <SelectField label="批准 Approval" value={options.file_editor_approval} onChange={(value) => update("file_editor_approval", value)} values={["manual", "auto", "readOnly"]} />
        <SelectField label="MCP" value={options.mcp_mode} onChange={(value) => update("mcp_mode", value)} values={["off", "auto"]} />
        <SelectField label="历史 History" value={options.history_mode} onChange={(value) => update("history_mode", value)} values={["off", "auto"]} />
        <SelectField label="自动化 Automation" value={options.automation_mode} onChange={(value) => update("automation_mode", value)} values={["off", "auto"]} />
      </div>
      <div className="checkGrid">
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_memory} onChange={(event) => update("rag_include_memory", event.target.checked)} /><span>RAG 记忆 MEMORY</span></label>
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_skills} onChange={(event) => update("rag_include_skills", event.target.checked)} /><span>RAG 技能 SKILL</span></label>
        <label className="checkLine"><input type="checkbox" checked={options.rag_include_knowledge} onChange={(event) => update("rag_include_knowledge", event.target.checked)} /><span>RAG 知识 KNOWLEDGE</span></label>
      </div>
      <label>
        <span>工具轮次 Tool Rounds</span>
        <input type="number" min="-1" value={options.max_tool_rounds} onChange={(event) => update("max_tool_rounds", Number(event.target.value))} />
        <small>-1 = 无限制，0 = 禁用工具</small>
      </label>
      <button className="secondaryButton" type="button" onClick={clearState} title="Clear conversation">
        <RefreshCw size={16} />
        <span>重置对话 Reset</span>
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

function StreamEvents({ events }) {
  const turns = [];
  let current = [];
  for (const event of events) {
    if (event.type === "user" && current.length > 0) {
      turns.push(current);
      current = [];
    }
    current.push(event);
  }
  if (current.length > 0) turns.push(current);
  return turns.map((turn, index) => <StreamTurn key={index} events={turn} />);
}

function StreamTurn({ events }) {
  const assistantIndex = findLastIndex(events, (event) => event.type === "assistant");
  const finalIndex = assistantIndex;
  const completed = finalIndex >= 0;
  const userEvents = events.filter((event) => event.type === "user");
  const finalEvent = completed ? events[finalIndex] : null;
  const operationEvents = events.filter((event, index) => index !== finalIndex && event.type !== "user");
  const currentWork = summarizeOperation(operationEvents[operationEvents.length - 1], completed);

  return (
    <div className="streamTurn">
      {userEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} />)}
      {operationEvents.length > 0 && (
        <details className="operationDetails">
          <summary>
            {!completed && <span className="runningDot" />}
            <span>{currentWork}</span>
            <small>{operationEvents.length} steps</small>
          </summary>
          {operationEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} compact />)}
        </details>
      )}
      {finalEvent && <StreamEvent event={finalEvent} />}
    </div>
  );
}

function StreamEvent({ event, compact = false }) {
  const type = event.type || "event";
  const fallback = event.text || event.query || event.url || "";
  const body = stringifyEventText(event, fallback);
  const images = extractLooseImages(body);
  return (
    <article className={`event event-${type}${compact ? " compact" : ""}`}>
      <div className="eventType">{eventLabel(type)}</div>
      <MarkdownText text={body} />
      {event.attachments?.length > 0 && <AttachmentList attachments={event.attachments} />}
      {images.length > 0 && <ImageStrip images={images} />}
      {type === "assistant" && <ReferenceList refs={extractReferences(body)} />}
    </article>
  );
}

function AttachmentList({ attachments }) {
  return (
    <div className="attachmentList">
      {attachments.map((item) => (
        <span key={item.path}>
          {item.content_type?.startsWith("image/") ? <Image size={14} /> : <Paperclip size={14} />}
          {item.filename || item.path}
        </span>
      ))}
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

function MarkdownText({ text }) {
  return (
    <div className="markdownBody">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
          img: ({ src, alt }) => <img src={normalizeImageSrc(src || "")} alt={alt || "image"} />,
        }}
      >
        {text || ""}
      </ReactMarkdown>
    </div>
  );
}

function ImageStrip({ images }) {
  return (
    <div className="imageStrip">
      {images.map((src) => <img key={src} src={normalizeImageSrc(src)} alt="tool output" loading="lazy" />)}
    </div>
  );
}

function DataView() {
  const [kind, setKind] = useState("instruction");
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState("");
  const [selectedMeta, setSelectedMeta] = useState({ writable: true, scope: "user" });
  const [content, setContent] = useState("");
  const [preview, setPreview] = useState(false);
  const [status, setStatus] = useState("");
  const [importName, setImportName] = useState("");
  const [importContent, setImportContent] = useState("# New Document\n\n");

  useEffect(() => {
    refreshResourceList();
  }, [kind]);

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
    } else {
      await fetchJson("/api/data/file", { method: "PUT", body: { path: selected, content } });
    }
    const ragStatus = await fetchJson("/api/rag/reindex", { method: "POST" }).catch(() => null);
    setStatus(ragStatus ? `已保存，向量索引已刷新：${ragStatus.chunk_count} 个片段` : "已保存");
    if (kind !== "instruction") await refreshListOnly();
  }

  async function renameSelected() {
    if (!selected || !selectedMeta.writable || !selected.toLowerCase().endsWith(".md")) return;
    const nextName = window.prompt("重命名 Markdown 文件", selected.split("/").pop() || "");
    if (!nextName) return;
    const data = await fetchJson("/api/data/file/rename", { method: "POST", body: { path: selected, new_name: nextName } });
    await refreshListOnly();
    await openFile(data.path);
    setStatus(`已重命名：${data.path}`);
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
    const data = await fetchJson("/api/data/import", { method: "POST", body: { kind, name: importName, content: importContent } });
    await refreshListOnly();
    await openFile(data.path);
    setImportName("");
    setStatus(`已导入：${data.path}`);
  }

  async function reindexRag() {
    const data = await fetchJson("/api/rag/reindex", { method: "POST" });
    setStatus(`向量索引已刷新：${data.chunk_count} 个片段`);
  }

  const resourceTabs = [
    ["instruction", "INSTRUCTION 指令"],
    ["memory", "MEMORY 记忆"],
    ["skills", "SKILL 技能"],
    ["knowledge", "KNOWLEDGE 知识"],
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
        <button className="secondaryButton full" onClick={reindexRag} type="button"><RefreshCw size={16} /><span>刷新 RAG</span></button>
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
            <span>{selectedMeta.writable ? "可编辑用户文件" : "系统级文件只读"}。指令是短规则，记忆是事实，技能是流程，知识是参考文档。</span>
          </div>
          <div className="rowActions">
            <button className="secondaryButton" onClick={() => setPreview((value) => !value)} disabled={!isMarkdown} type="button"><FileText size={16} /><span>{preview ? "编辑" : "显示 MD"}</span></button>
            <button className="secondaryButton" onClick={renameSelected} disabled={!selectedMeta.writable || !isMarkdown} type="button"><AtSign size={16} /><span>重命名</span></button>
            <button className="primaryButton" onClick={saveFile} disabled={!selected || !selectedMeta.writable} type="button"><Save size={16} /><span>保存</span></button>
          </div>
        </div>
        {preview ? (
          <div className="markdownPreview"><MarkdownText text={content} /></div>
        ) : (
          <textarea className="codeEditor" value={content} onChange={(event) => setContent(event.target.value)} readOnly={!selectedMeta.writable} placeholder="选择并编辑指令、记忆、技能或知识文件。" />
        )}
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
      <form className="importPane" onSubmit={importResource}>
        <h2>导入 IMPORT</h2>
        <label><span>名称 Name</span><input value={importName} onChange={(event) => setImportName(event.target.value)} placeholder={kind === "skills" ? "my-skill" : "notes.md"} /></label>
        <label><span>内容 Content</span><textarea value={importContent} onChange={(event) => setImportContent(event.target.value)} /></label>
        <button className="primaryButton" type="submit"><Plus size={16} /><span>导入到当前分类</span></button>
      </form>
    </section>
  );
}

function ConfigView({ onSaved }) {
  const [config, setConfig] = useState({ providers: {} });
  const [selectedProvider, setSelectedProvider] = useState("");
  const [status, setStatus] = useState("");

  useEffect(() => {
    refreshConfig();
  }, []);

  async function refreshConfig() {
    const data = await fetchJson("/api/config");
    const nextConfig = data.config || { providers: {} };
    setConfig(nextConfig);
    setSelectedProvider(nextConfig.default_provider || Object.keys(nextConfig.providers || {})[0] || "");
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
      draft.providers[name] ||= { base_url: "", api_key_env: "", api_key: "", models: [] };
      draft.default_provider ||= name;
      return draft;
    });
    setSelectedProvider(name);
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

  function setDefaultProvider() {
    if (!selectedProvider) return;
    updateConfig((draft) => ({ ...draft, default_provider: selectedProvider }));
  }

  async function saveConfig() {
    try {
      await fetchJson("/api/config", { method: "PUT", body: { config } });
      setStatus("模型 API 配置已保存");
      onSaved?.();
    } catch (error) {
      setStatus(`保存失败: ${String(error.message || error)}`);
    }
  }

  const providers = config.providers || {};
  const provider = selectedProvider ? providers[selectedProvider] || {} : {};
  const modelLines = Array.isArray(provider.models)
    ? provider.models.map((item) => (typeof item === "string" ? item : item.alias || item.id || "")).join("\n")
    : "";

  return (
    <section className="configLayout">
      <aside className="providerPane">
        <div className="editorHeader">
          <h2>供应商 PROVIDER</h2>
          <button className="iconButton neutral" onClick={addProvider} type="button" title="新增"><Plus size={16} /></button>
        </div>
        {Object.keys(providers).map((name) => (
          <button key={name} className={selectedProvider === name ? "providerItem active" : "providerItem"} onClick={() => setSelectedProvider(name)} type="button">
            <span><Key size={15} />{name}</span>
            {config.default_provider === name && <small>DEFAULT</small>}
          </button>
        ))}
      </aside>
      <div className="modelEditor">
        <div className="editorHeader">
          <div>
            <h2>模型 API 配置 / MODEL API CONFIG</h2>
            <span>选择左侧供应商后填写 Base URL、API Key 或环境变量，以及模型列表。</span>
          </div>
          <div className="rowActions">
            <button className="secondaryButton" onClick={setDefaultProvider} disabled={!selectedProvider} type="button"><Check size={16} /><span>设为默认</span></button>
            <button className="secondaryButton" onClick={renameProvider} disabled={!selectedProvider} type="button"><AtSign size={16} /><span>重命名</span></button>
            <button className="dangerButton" onClick={deleteProvider} disabled={!selectedProvider} type="button"><Trash2 size={16} /><span>删除</span></button>
            <button className="secondaryButton" onClick={refreshConfig} type="button"><RefreshCw size={16} /><span>刷新</span></button>
            <button className="primaryButton" onClick={saveConfig} type="button"><Save size={16} /><span>保存</span></button>
          </div>
        </div>
        {selectedProvider ? (
          <div className="modelForm">
            <label><span>当前供应商 Current Provider</span><input value={selectedProvider} readOnly /></label>
            <label><span>默认模型 Default Model</span><input value={config.default_model || ""} onChange={(event) => updateConfig((draft) => ({ ...draft, default_model: event.target.value }))} placeholder={`${selectedProvider}:deepseek-chat`} /></label>
            <label><span>Base URL</span><input value={provider.base_url || ""} onChange={(event) => updateProvider("base_url", event.target.value)} placeholder="https://api.deepseek.com/v1" /></label>
            <label><span>API Key</span><input value={provider.api_key || ""} onChange={(event) => updateProvider("api_key", event.target.value)} placeholder="直接保存 API Key" /></label>
            <label><span>API Key Env</span><input value={provider.api_key_env || ""} onChange={(event) => updateProvider("api_key_env", event.target.value)} placeholder="DEEPSEEK_API_KEY" /></label>
            <label><span>默认供应商 Default Provider</span><input value={config.default_provider || ""} readOnly /></label>
            <label className="fullWidth"><span>模型列表 Models，每行一个</span><textarea value={modelLines} onChange={(event) => updateProvider("models", event.target.value.split("\n").map((line) => line.trim()).filter(Boolean))} placeholder={"deepseek-chat\ndeepseek-reasoner"} /></label>
          </div>
        ) : (
          <div className="emptyState">左侧新增或选择一个 Provider。</div>
        )}
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
    </section>
  );
}

function SystemView({ theme, setTheme, language, setLanguage }) {
  return (
    <section className="systemLayout">
      <div className="systemPanel">
        <div className="editorHeader">
          <div>
            <h2>系统设置 / SYSTEM SETTINGS</h2>
            <span>这些设置只影响界面，不会混入当前对话提示词。</span>
          </div>
        </div>
        <div className="modelForm">
          <SelectField label="主题 Theme" value={theme} onChange={setTheme} values={["system", "light", "dark"]} icon={<Sun size={14} />} />
          <SelectField label="语言 Language" value={language} onChange={setLanguage} values={["zh", "en", "both"]} icon={<Languages size={14} />} />
        </div>
      </div>
    </section>
  );
}

function McpView() {
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
      setStatus(`Load failed: ${String(error.message || error)}`);
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
      setStatus("Name is required");
      return;
    }
    try {
      const data = await fetchJson(`/api/mcp/servers/${encodeURIComponent(payload.name)}`, { method: "PUT", body: payload });
      setConfig(data);
      const saved = data.servers?.[payload.name] || payload;
      editServer(payload.name, saved);
      setStatus(`Saved ${payload.name}`);
    } catch (error) {
      setStatus(`Save failed: ${String(error.message || error)}`);
    }
  }

  async function testCurrentServer() {
    const payload = buildMcpPayload(form);
    if (payload.transport === "streamable_http" && !payload.url) {
      setStatus("Test failed: url is required");
      return;
    }
    if (payload.transport === "stdio" && !payload.command) {
      setStatus("Test failed: command is required");
      return;
    }
    try {
      const data = await fetchJson("/api/mcp/test", { method: "POST", body: payload });
      setStatus(`Test ok: ${summarizeMcpTest(data)}`);
    } catch (error) {
      setStatus(`Test failed: ${String(error.message || error)}`);
    }
  }

  async function testSavedServer(name) {
    try {
      const data = await fetchJson(`/api/mcp/servers/${encodeURIComponent(name)}/test`, { method: "POST" });
      setStatus(`Test ok: ${summarizeMcpTest(data)}`);
    } catch (error) {
      setStatus(`Test failed: ${String(error.message || error)}`);
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
      setStatus(`Deleted ${name}`);
      await refresh();
    } catch (error) {
      setStatus(`Delete failed: ${String(error.message || error)}`);
    }
  }

  const servers = useMemo(() => Object.entries(config.servers || {}), [config]);
  return (
    <section className="mcpLayout">
      <div className="serverListPane">
        <div className="editorHeader"><h2>MCP SERVERS</h2><button className="secondaryButton" onClick={refresh} type="button"><RefreshCw size={16} /><span>Refresh</span></button></div>
        {servers.map(([name, server]) => (
          <article className="serverItem" key={name}>
            <div><strong>{name}</strong><span>{server.enabled ? "enabled" : "disabled"}</span></div>
            <code>{server.transport || "stdio"} {server.url || server.command || ""}</code>
            <div className="rowActions">
              <button className="secondaryButton" onClick={() => editServer(name, server)} type="button"><FileText size={15} /><span>Edit</span></button>
              <button className="secondaryButton" onClick={() => testSavedServer(name)} type="button"><Plug size={15} /><span>Test</span></button>
              <button className="dangerButton" onClick={() => deleteServer(name)} type="button"><Trash2 size={15} /><span>Delete</span></button>
            </div>
          </article>
        ))}
      </div>
      <form className="mcpForm" onSubmit={saveServer}>
        <h2>ADD OR EDIT SERVER</h2>
        <label><span>Name</span><input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="rent" /></label>
        <SelectField label="Transport" value={form.transport} onChange={(value) => setForm({ ...form, transport: value })} values={["streamable_http", "stdio"]} />
        {form.transport === "stdio" ? (
          <>
            <label><span>Command</span><input value={form.command} onChange={(event) => setForm({ ...form, command: event.target.value })} placeholder="node" /></label>
            <label><span>Args 每行一个</span><textarea value={form.argsText} onChange={(event) => setForm({ ...form, argsText: event.target.value })} placeholder={"server.js\n--port\n5050"} /></label>
            <KeyValueEditor title="Env" rows={form.envRows} setForm={setForm} field="envRows" />
          </>
        ) : (
          <>
            <label><span>URL</span><input value={form.url} onChange={(event) => setForm({ ...form, url: event.target.value })} placeholder="http://127.0.0.1:5050/mcp" /></label>
            <KeyValueEditor title="Headers" rows={form.headerRows} setForm={setForm} field="headerRows" />
          </>
        )}
        <label><span>Timeout</span><input type="number" min="1" value={form.timeout} onChange={(event) => setForm({ ...form, timeout: event.target.value })} /></label>
        <label><span>SSE Read Timeout</span><input type="number" min="1" value={form.sse_read_timeout} onChange={(event) => setForm({ ...form, sse_read_timeout: event.target.value })} /></label>
        <label className="checkLine"><input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} /><span>Enabled</span></label>
        <div className="formActions">
          <button className="primaryButton" type="submit"><Save size={16} /><span>Save Server</span></button>
          <button className="secondaryButton" type="button" onClick={testCurrentServer}><Plug size={16} /><span>Test Connection</span></button>
        </div>
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </form>
    </section>
  );
}

function KeyValueEditor({ title, rows, setForm, field }) {
  return (
    <div className="headerEditor">
      <div className="labelText">{title}</div>
      {rows.map((row, index) => (
        <div className="headerRow" key={index}>
          <input value={row.key} onChange={(event) => updateHeaderRow(index, "key", event.target.value, setForm, field)} placeholder="Key" />
          <input value={row.value} onChange={(event) => updateHeaderRow(index, "value", event.target.value, setForm, field)} placeholder="Value" />
          <button className="iconButton" type="button" onClick={() => removeHeaderRow(index, setForm, field)} title={`Remove ${title}`}>
            <Trash2 size={15} />
          </button>
        </div>
      ))}
      <button className="secondaryButton" type="button" onClick={() => addHeaderRow(setForm, field)}><Plus size={16} /><span>Add {title}</span></button>
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
  return { ...options, model: options.model || null, system_prompt: options.system_prompt || null };
}

function stringifyEventText(event, fallback) {
  if (["tool_call", "assistant_progress", "approval_required", "error", "user", "assistant"].includes(event.type)) {
    return fallback;
  }
  return JSON.stringify(event, null, 2);
}

function summarizeOperation(event, completed) {
  if (!event) return completed ? "已完成" : "正在准备";
  const name = event.tool || event.type || "step";
  const detail = event.query || event.url || event.text || "";
  const clean = String(detail).replace(/\s+/g, " ").slice(0, 90);
  return `${completed ? "已执行" : "正在"} ${eventLabel(name)}${clean ? `: ${clean}` : ""}`;
}

function eventLabel(type) {
  const labels = {
    user: "USER",
    assistant: "ASSISTANT",
    assistant_progress: "THINKING",
    tool_call: "TOOL",
    approval_required: "APPROVAL",
    error: "ERROR",
    python: "PYTHON",
    rag: "RAG",
    mcp: "MCP",
    curl: "HTTP",
    webSearch: "WEB",
    fileEditor: "FILE",
    settings_changed: "SETTINGS",
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
  for (const match of text.matchAll(/(?:data|backend)\/[^\s)]+\.(?:md|txt|json|yaml|yml|png|jpg|jpeg|gif|webp|svg)/gi)) {
    refs.push(cleanImageSrc(match[0]));
  }
  return Array.from(new Set(refs)).slice(0, 8);
}

function normalizeReferenceHref(ref) {
  if (/^https?:\/\//i.test(ref)) return ref;
  if (/\.(?:png|jpg|jpeg|gif|webp|svg)$/i.test(ref)) return normalizeImageSrc(ref);
  return "#";
}

function cleanImageSrc(src) {
  return String(src || "").trim().replace(/[.,;:]+$/g, "");
}

function normalizeImageSrc(src) {
  if (!src) return "";
  if (/^https?:\/\//i.test(src) || src.startsWith("data:")) return src;
  if (src.startsWith("/api/")) return src;
  return `${API_BASE}/api/artifact?path=${encodeURIComponent(src.replaceAll("\\", "/"))}`;
}

function toolMentionOptions() {
  return [
    { label: "工具 Python", token: "@tool:python" },
    { label: "工具 RAG", token: "@tool:rag" },
    { label: "工具 Web", token: "@tool:webSearch" },
    { label: "工具 HTTP", token: "@tool:curl" },
    { label: "工具 文件编辑", token: "@tool:fileEditor" },
    { label: "工具 MCP", token: "@tool:mcp" },
    { label: "工具 历史", token: "@tool:history" },
  ].concat([{ label: "Tool Settings", token: "@tool:settings" }]);
}

function conversationMentionOptions(conversations) {
  return conversations.map((item) => ({
    label: `历史 ${item.title || item.id}`,
    token: `@history:${item.id}`,
  }));
}

function fileMentionOptions(kind, files) {
  const labelMap = { memory: "记忆", skill: "技能", knowledge: "知识", instruction: "指令" };
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


createRoot(document.getElementById("root")).render(<App />);

