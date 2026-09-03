import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Database, FileText, Plug, Plus, RefreshCw, Save, Send, Trash2 } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const emptyOptions = {
  model: "",
  system_prompt: "",
  web_search_mode: "auto",
  web_search_provider: "duckduckgo",
  rag_mode: "auto",
  curl_mode: "auto",
  python_mode: "auto",
  file_editor_mode: "auto",
  file_editor_approval: "auto",
  mcp_mode: "auto",
  max_tool_rounds: 20,
};

function App() {
  const [tab, setTab] = useState("chat");
  const [models, setModels] = useState([]);
  const [options, setOptions] = useState(emptyOptions);

  useEffect(() => {
    fetchJson("/api/models")
      .then((data) => {
        setModels(data.models || []);
        setOptions((current) => ({ ...current, model: data.defaultModel || current.model }));
      })
      .catch(() => setModels([]));
  }, []);

  return (
    <main className="appShell">
      <header className="topbar">
        <div>
          <h1>AI Agent Tester</h1>
          <p>Chat, tools, RAG, skills, memory, and MCP in one local panel.</p>
        </div>
        <nav className="tabs" aria-label="Main views">
          <TabButton active={tab === "chat"} onClick={() => setTab("chat")} icon={<Send size={16} />} label="Chat" />
          <TabButton active={tab === "data"} onClick={() => setTab("data")} icon={<Database size={16} />} label="Data" />
          <TabButton active={tab === "mcp"} onClick={() => setTab("mcp")} icon={<Plug size={16} />} label="MCP" />
        </nav>
      </header>

      {tab === "chat" && <ChatView models={models} options={options} setOptions={setOptions} />}
      {tab === "data" && <DataView />}
      {tab === "mcp" && <McpView />}
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

function ChatView({ models, options, setOptions }) {
  const [message, setMessage] = useState("");
  const [events, setEvents] = useState([]);
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState(false);
  const outputRef = useRef(null);

  useEffect(() => {
    outputRef.current?.scrollTo({ top: outputRef.current.scrollHeight });
  }, [events]);

  async function sendMessage(event) {
    event.preventDefault();
    if (!message.trim() || busy) return;
    setBusy(true);
    setEvents((items) => [...items, { type: "user", text: message }]);
    const nextMessage = message;
    setMessage("");

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: nextMessage, state, options: normalizeOptions(options) }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      if (!response.body) throw new Error("HTTP response has no stream body");
      await readSse(response.body, (eventData) => {
        setEvents((items) => [...items, eventData]);
        if (eventData.type === "assistant" && eventData.state) setState(eventData.state);
      });
    } catch (error) {
      setEvents((items) => [...items, { type: "error", text: String(error.message || error) }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chatLayout">
      <div className="chatPane">
        <div className="stream" ref={outputRef}>
          {events.length === 0 && <div className="emptyState">Send a message to test streaming tool events.</div>}
          <StreamEvents events={events} />
        </div>
        <form className="composer" onSubmit={sendMessage}>
          <textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ask the agent to search, use RAG, run Python, edit a file, or call MCP..." />
          <button className="primaryButton" type="submit" disabled={busy || !message.trim()} title="Send">
            <Send size={18} />
            <span>{busy ? "Running" : "Send"}</span>
          </button>
        </form>
      </div>
      <SettingsPanel models={models} options={options} setOptions={setOptions} clearState={() => { setState(null); setEvents([]); }} />
    </section>
  );
}

function SettingsPanel({ models, options, setOptions, clearState }) {
  const update = (key, value) => setOptions((current) => ({ ...current, [key]: value }));
  return (
    <aside className="settingsPane">
      <h2>Run Settings</h2>
      <label>
        <span>Model</span>
        <select value={options.model} onChange={(event) => update("model", event.target.value)}>
          <option value="">Default</option>
          {models.map((model) => <option key={model.value} value={model.value}>{model.label}</option>)}
        </select>
      </label>
      <label>
        <span>Instruction</span>
        <textarea className="smallTextArea" value={options.system_prompt} onChange={(event) => update("system_prompt", event.target.value)} placeholder="Optional system prompt override" />
      </label>
      <div className="settingGrid">
        <SelectField label="Web" value={options.web_search_mode} onChange={(value) => update("web_search_mode", value)} values={["off", "auto"]} />
        <SelectField label="Web Provider" value={options.web_search_provider} onChange={(value) => update("web_search_provider", value)} values={["duckduckgo", "searxng", "tavily"]} />
        <SelectField label="RAG" value={options.rag_mode} onChange={(value) => update("rag_mode", value)} values={["off", "on", "auto"]} />
        <SelectField label="Curl" value={options.curl_mode} onChange={(value) => update("curl_mode", value)} values={["off", "auto"]} />
        <SelectField label="Python" value={options.python_mode} onChange={(value) => update("python_mode", value)} values={["off", "auto"]} />
        <SelectField label="File Edit" value={options.file_editor_mode} onChange={(value) => update("file_editor_mode", value)} values={["off", "auto"]} />
        <SelectField label="Approval" value={options.file_editor_approval} onChange={(value) => update("file_editor_approval", value)} values={["manual", "auto", "readOnly"]} />
        <SelectField label="MCP" value={options.mcp_mode} onChange={(value) => update("mcp_mode", value)} values={["off", "auto"]} />
      </div>
      <label>
        <span>Max Tool Rounds</span>
        <input type="number" min="1" max="20" value={options.max_tool_rounds} onChange={(event) => update("max_tool_rounds", Number(event.target.value))} />
      </label>
      <button className="secondaryButton" type="button" onClick={clearState} title="Clear conversation">
        <RefreshCw size={16} />
        <span>Reset</span>
      </button>
    </aside>
  );
}

function SelectField({ label, value, onChange, values }) {
  return (
    <label>
      <span>{label}</span>
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
  const finalIndex = findLastIndex(events, (event) => event.type === "assistant" || event.type === "error");
  const completed = finalIndex >= 0;
  const primaryEvents = completed
    ? events.filter((event, index) => event.type === "user" || index === finalIndex)
    : events.filter((event) => event.type === "user");
  const operationEvents = completed
    ? events.filter((event, index) => index !== finalIndex && event.type !== "user")
    : events.filter((event) => event.type !== "user");

  return (
    <div className="streamTurn">
      {primaryEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} />)}
      {operationEvents.length > 0 && (
        <details className="operationDetails" open={!completed}>
          <summary>{completed ? `Operations (${operationEvents.length})` : "Operations"}</summary>
          {operationEvents.map((event, index) => <StreamEvent key={`${event.type}-${index}`} event={event} compact />)}
        </details>
      )}
    </div>
  );
}

function StreamEvent({ event, compact = false }) {
  const type = event.type || "event";
  const fallback = event.text || event.query || event.url || "";
  const body = stringifyEventText(event, fallback);
  const images = extractImages(body);
  return (
    <article className={`event event-${type}${compact ? " compact" : ""}`}>
      <div className="eventType">{type}</div>
      <MarkdownText text={body} />
      {images.length > 0 && <ImageStrip images={images} />}
    </article>
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

function stringifyEventText(event, fallback) {
  if (["tool_call", "assistant_progress", "approval_required", "error", "user", "assistant"].includes(event.type)) {
    return fallback;
  }
  return JSON.stringify(event, null, 2);
}

function extractImages(text) {
  const patterns = [
    /!\[[^\]]*]\(([^)]+)\)/g,
    /image:\s*(https?:\/\/[^\s<>"')]+)/gi,
    /(https?:\/\/[^\s<>"')]+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?[^\s<>"')]+)?)/gi,
    /([A-Za-z]:[\\/][^\n\r"'<>|]+\bpython_runs[\\/][^\n\r"'<>|]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\/runtime\/python_runs\/[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
    /(backend\\runtime\\python_runs\\[^\s)]+\.(?:png|jpg|jpeg|gif|webp|svg))/gi,
  ];
  const found = [];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) found.push(cleanImageSrc(match[1]));
  }
  return Array.from(new Set(found.filter(Boolean)));
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

function DataView() {
  const [kind, setKind] = useState("skills");
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState("");
  const [content, setContent] = useState("");
  const [status, setStatus] = useState("");
  const [skillName, setSkillName] = useState("");
  const [skillContent, setSkillContent] = useState("# New Skill\n\nUse this skill when...\n");

  useEffect(() => { refreshFiles(); }, [kind]);

  async function refreshFiles() {
    const data = await fetchJson(`/api/data/files?kind=${kind}`);
    setFiles(data.files || []);
    setStatus("");
  }

  async function openFile(path) {
    setSelected(path);
    const data = await fetchJson(`/api/data/file?path=${encodeURIComponent(path)}`);
    setContent(data.content || "");
  }

  async function saveFile() {
    if (!selected) return;
    await fetchJson("/api/data/file", { method: "PUT", body: { path: selected, content } });
    setStatus(`Saved ${selected}`);
    refreshFiles();
  }

  async function importSkill(event) {
    event.preventDefault();
    if (!skillName.trim()) return;
    const data = await fetchJson("/api/skills/import", { method: "POST", body: { name: skillName, content: skillContent } });
    setKind("skills");
    setSkillName("");
    setStatus(`Imported ${data.path}`);
    await refreshFiles();
    await openFile(data.path);
  }

  return (
    <section className="dataLayout">
      <aside className="fileListPane">
        <div className="segmented">
          {["skills", "memory", "knowledge"].map((item) => <button key={item} className={kind === item ? "active" : ""} onClick={() => setKind(item)} type="button">{item}</button>)}
        </div>
        <button className="secondaryButton full" onClick={refreshFiles} type="button"><RefreshCw size={16} /><span>Refresh</span></button>
        <div className="fileList">
          {files.map((file) => <button key={file} className={selected === file ? "fileItem active" : "fileItem"} onClick={() => openFile(file)} type="button"><FileText size={15} /><span>{file}</span></button>)}
        </div>
      </aside>
      <div className="editorPane">
        <div className="editorHeader">
          <h2>{selected || "Select a file"}</h2>
          <button className="primaryButton" onClick={saveFile} disabled={!selected} type="button"><Save size={16} /><span>Save</span></button>
        </div>
        <textarea className="codeEditor" value={content} onChange={(event) => setContent(event.target.value)} placeholder="Edit skill, memory, or knowledge text here." />
        {status && <p className="statusLine"><Check size={15} />{status}</p>}
      </div>
      <form className="importPane" onSubmit={importSkill}>
        <h2>Import Skill</h2>
        <label><span>Name</span><input value={skillName} onChange={(event) => setSkillName(event.target.value)} placeholder="my-skill" /></label>
        <label><span>SKILL.md</span><textarea value={skillContent} onChange={(event) => setSkillContent(event.target.value)} /></label>
        <button className="primaryButton" type="submit"><Plus size={16} /><span>Import</span></button>
      </form>
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
    timeout: 5,
    sse_read_timeout: 300,
  });
  const [status, setStatus] = useState("");

  useEffect(() => { refresh(); }, []);

  async function refresh() {
    try {
      const data = await fetchJson("/api/mcp/servers");
      setConfig(data);
    } catch (error) {
      setStatus(`Load failed: ${String(error.message || error)}`);
    }
  }

  function editServer(name, server) {
    setForm({
      name,
      enabled: Boolean(server.enabled),
      transport: server.transport || "streamable_http",
      url: server.url || "",
      headerRows: headerRowsFromObject(server.headers),
      timeout: Number(server.timeout || 5),
      sse_read_timeout: Number(server.sse_read_timeout || 300),
    });
  }

  async function saveServer(event) {
    event.preventDefault();
    const payload = buildMcpPayload(form);
    if (!payload.name) {
      setStatus("Save failed: name is required");
      return;
    }
    if (!payload.url) {
      setStatus("Save failed: url is required");
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
    if (!payload.url) {
      setStatus("Test failed: url is required");
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
    const url = cleanInputUrl(formValue.url);
    if (!formValue.name.trim()) {
      return {
        name: "",
        enabled: formValue.enabled,
        transport: "streamable_http",
        url,
        headers: headersObject(formValue.headerRows),
        timeout: Number(formValue.timeout),
        sse_read_timeout: Number(formValue.sse_read_timeout),
      };
    }
    return {
      name: formValue.name.trim(),
      enabled: formValue.enabled,
      transport: "streamable_http",
      url,
      headers: headersObject(formValue.headerRows),
      timeout: Number(formValue.timeout),
      sse_read_timeout: Number(formValue.sse_read_timeout),
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
        <div className="editorHeader"><h2>MCP Servers</h2><button className="secondaryButton" onClick={refresh} type="button"><RefreshCw size={16} /><span>Refresh</span></button></div>
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
        <h2>Add or Edit Server</h2>
        <label><span>Name</span><input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="rent" /></label>
        <label><span>URL</span><input value={form.url} onChange={(event) => setForm({ ...form, url: event.target.value })} placeholder="http://127.0.0.1:5050/mcp" /></label>
        <div className="headerEditor">
          <div className="labelText">Headers</div>
          {form.headerRows.map((row, index) => (
            <div className="headerRow" key={index}>
              <input value={row.key} onChange={(event) => updateHeaderRow(index, "key", event.target.value, setForm)} placeholder="Key" />
              <input value={row.value} onChange={(event) => updateHeaderRow(index, "value", event.target.value, setForm)} placeholder="Value" />
              <button className="iconButton" type="button" onClick={() => removeHeaderRow(index, setForm)} title="Remove header"><Trash2 size={15} /></button>
            </div>
          ))}
          <button className="secondaryButton" type="button" onClick={() => addHeaderRow(setForm)}><Plus size={16} /><span>Add Header</span></button>
        </div>
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

function updateHeaderRow(index, field, value, setForm) {
  setForm((current) => {
    const nextRows = current.headerRows.map((row, rowIndex) => (
      rowIndex === index ? { ...row, [field]: value } : row
    ));
    return { ...current, headerRows: nextRows };
  });
}

function addHeaderRow(setForm) {
  setForm((current) => ({
    ...current,
    headerRows: [...current.headerRows, { key: "", value: "" }],
  }));
}

function removeHeaderRow(index, setForm) {
  setForm((current) => {
    const nextRows = current.headerRows.filter((_, rowIndex) => rowIndex !== index);
    return { ...current, headerRows: nextRows.length > 0 ? nextRows : [{ key: "", value: "" }] };
  });
}

function findLastIndex(items, predicate) {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (predicate(items[index], index)) return index;
  }
  return -1;
}

createRoot(document.getElementById("root")).render(<App />);
