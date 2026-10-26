import React, { useEffect, useState } from "react";
import { Check, Database, Paperclip, RefreshCw, X } from "lucide-react";
import { Segmented } from "./ui.jsx";

// What a refresh did, as one line (chunks, reused files, LLM calls for chunking and for chunk context, fallbacks).
export function formatRagRefreshStatus(prefix, ragStatus) {
  if (!ragStatus) return prefix;
  const details = [`${ragStatus.chunk_count || 0} 个片段`];
  if (ragStatus.documents_reused) details.push(`复用 ${ragStatus.documents_reused} 个文件`);
  if (ragStatus.llm_calls) details.push(`LLM 切分 ${ragStatus.llm_calls} 次 / 约 ${ragStatus.estimated_input_tokens || 0} 输入 token`);
  if (ragStatus.llm_fallbacks) details.push(`回退 ${ragStatus.llm_fallbacks} 次`);
  if (ragStatus.context_calls) details.push(`上下文 ${ragStatus.context_calls} 次 / 约 ${ragStatus.context_input_tokens || 0} 输入 token`);
  if (ragStatus.context_reused) details.push(`复用上下文 ${ragStatus.context_reused} 个`);
  if (ragStatus.context_pending) details.push(`还有 ${ragStatus.context_pending} 个片段没有上下文（再刷新一次继续）`);
  if (ragStatus.embedded_texts) details.push(`向量 ${ragStatus.embedded_texts} 个`);
  if (ragStatus.embedding_fallback) details.push(`向量模型不可用，已用本地向量：${ragStatus.embedding_fallback}`);
  return `${prefix}：${details.join("，")}`;
}

const DEFAULT_INGESTION = { split_mode: "simple", model: "", contextual: "off", context_model: "", embedding_model: "" };
const DEFAULT_SEARCH = { rerank: "off", rerank_model: "", default_method: "hybrid", require_citations: true };

function ModelSelect({ value, models, onChange, disabled, emptyLabel }) {
  return (
    <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
      <option value="">{emptyLabel}</option>
      {(models || []).map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
    </select>
  );
}

// RAG settings of the Data page: how documents are split, whether an LLM writes a context for each chunk, which
// vectors are used, how a search ranks, plus refresh and upload. Settings are saved as soon as they change; the
// backend reads them again for every refresh and every search.
export default function RagPanel({ text, models, request, onUpload, uploading }) {
  const [ingestion, setIngestion] = useState(DEFAULT_INGESTION);
  const [search, setSearch] = useState(DEFAULT_SEARCH);
  const [refresh, setRefresh] = useState({ phase: "idle", message: "" });
  const loading = refresh.phase === "loading";

  useEffect(() => {
    request("/api/settings").then((data) => {
      setIngestion({ ...DEFAULT_INGESTION, ...(data.settings?.rag_ingestion || {}) });
      setSearch({ ...DEFAULT_SEARCH, ...(data.settings?.rag_search || {}) });
    }).catch(() => {});
  }, []);

  async function save(section, patch) {
    const apply = section === "rag_ingestion" ? setIngestion : setSearch;
    apply((current) => ({ ...current, ...patch }));
    try {
      await request("/api/settings", { method: "PATCH", body: { patch: { [section]: patch } } });
    } catch (error) {
      setRefresh({ phase: "error", message: `${text("保存失败", "Save failed")}: ${String(error.message || error)}` });
    }
  }

  async function reindex() {
    if (loading) return;
    setRefresh({ phase: "loading", message: text("正在刷新 RAG 索引…", "Refreshing RAG index…") });
    try {
      const data = await request("/api/rag/reindex", { method: "POST", body: {} });
      setRefresh({ phase: "success", message: formatRagRefreshStatus(text("RAG 刷新成功", "RAG refresh succeeded"), data) });
    } catch (error) {
      setRefresh({ phase: "error", message: `${text("RAG 刷新失败", "RAG refresh failed")}: ${String(error.message || error)}` });
    }
  }

  const contextual = ingestion.contextual === "llm";
  const reranking = search.rerank !== "off";
  return (
    <section className="dataSidebarSection" aria-label={text("RAG 设置", "RAG settings")}>
      <div className="dataSidebarHeading">
        <Database size={15} />
        <span>{text("RAG 索引", "RAG INDEX")}</span>
      </div>
      <div className="ragField">
        <span>{text("切分方式", "Chunking")}</span>
        <div className="ragSegment">
          <Segmented
            ariaLabel={text("切分方式", "Chunking")}
            value={ingestion.split_mode}
            onChange={(value) => save("rag_ingestion", { split_mode: value })}
            options={[{ value: "simple", label: text("简单", "Simple") }, { value: "llm", label: "LLM" }]}
          />
        </div>
        {ingestion.split_mode === "llm" && (
          <ModelSelect value={ingestion.model} models={models} disabled={loading} emptyLabel={text("默认模型", "Default model")} onChange={(value) => save("rag_ingestion", { model: value })} />
        )}
      </div>
      <div className="ragField">
        <span>{text("分块上下文", "Chunk context")}</span>
        <div className="ragSegment">
          <Segmented
            ariaLabel={text("分块上下文", "Chunk context")}
            value={ingestion.contextual}
            onChange={(value) => save("rag_ingestion", { contextual: value })}
            options={[{ value: "off", label: text("关闭", "Off") }, { value: "llm", label: "LLM" }]}
          />
        </div>
        {contextual && (
          <ModelSelect value={ingestion.context_model} models={models} disabled={loading} emptyLabel={text("默认模型", "Default model")} onChange={(value) => save("rag_ingestion", { context_model: value })} />
        )}
        <p className="ragHint">
          {contextual
            ? text("每个片段让模型结合整篇文档写一两句背景，和片段一起做关键词与向量索引。每个片段一次调用，结果会缓存，只为新增内容付费。", "A model writes one or two sentences of background for each chunk from the whole document; it is indexed with the chunk for keywords and vectors. One call per chunk, cached, so only new text costs.")
            : text("开启后，找不到“这是谁的、哪一节的”这类缺少背景的片段会更容易被检索到。", "Turn on to make chunks that lack their background (whose, which section) easier to find.")}
        </p>
      </div>
      <div className="ragField">
        <span>{text("向量模型", "Vector model")}</span>
        <ModelSelect
          value={ingestion.embedding_model}
          models={models}
          disabled={loading}
          emptyLabel={text("本地字符 n-gram（无需模型）", "Local character n-grams (no model)")}
          onChange={(value) => save("rag_ingestion", { embedding_model: value })}
        />
        <p className="ragHint">
          {ingestion.embedding_model
            ? text("用该供应商的 /embeddings 接口生成向量；调用失败会自动退回本地向量。", "Vectors come from that provider's /embeddings endpoint; a failure falls back to the local vectors.")
            : text("本地向量只匹配字面相近的词。要匹配同义表达，请先在「模型」页给供应商添加嵌入模型，再在这里选择。", "Local vectors only match similar-looking words. To match paraphrases, add an embedding model to a provider on the Models page and pick it here.")}
        </p>
      </div>
      <div className="dataSidebarHeading ragSubheading">
        <span>{text("检索", "SEARCH")}</span>
      </div>
      <label className="ragField">
        <span>{text("默认检索方式", "Default method")}</span>
        <select value={search.default_method} disabled={loading} onChange={(event) => save("rag_search", { default_method: event.target.value })}>
          <option value="hybrid">{text("混合（BM25 + 向量）", "Hybrid (BM25 + vector)")}</option>
          <option value="bm25">{text("仅关键词 BM25", "BM25 keywords only")}</option>
          <option value="vector">{text("仅向量", "Vector only")}</option>
        </select>
        <small>{text("模型每次检索也可以自己选。", "The model can also choose per search.")}</small>
      </label>
      <div className="ragField">
        <span>{text("重排", "Rerank")}</span>
        <div className="ragSegment">
          <Segmented
            ariaLabel={text("重排", "Rerank")}
            value={search.rerank}
            onChange={(value) => save("rag_search", { rerank: value })}
            options={[{ value: "off", label: text("关闭", "Off") }, { value: "llm", label: "LLM" }, { value: "api", label: "API" }]}
          />
        </div>
        {reranking && (
          <ModelSelect
            value={search.rerank_model}
            models={models}
            disabled={loading}
            emptyLabel={search.rerank === "llm" ? text("默认模型", "Default model") : text("选择重排模型", "Choose a rerank model")}
            onChange={(value) => save("rag_search", { rerank_model: value })}
          />
        )}
        <p className="ragHint">
          {search.rerank === "llm" && text("每次检索多一次模型调用，让模型逐条打分后重排前 12 条，更准但更慢。", "One extra model call per search scores the top 12 results and reorders them: more precise, slower.")}
          {search.rerank === "api" && text("调用供应商的 /rerank 接口（Cohere、Jina、SiliconFlow 等）。需要选择重排模型。", "Calls the provider's /rerank endpoint (Cohere, Jina, SiliconFlow…). A rerank model is required.")}
          {search.rerank === "off" && text("不重排。模型也可以在单次检索时要求重排。自动注入的上下文永远不重排。", "No reranking. The model can still ask for it on one search. Automatic context is never reranked.")}
        </p>
      </div>
      <div className="ragField">
        <span>{text("引用检查", "Citation check")}</span>
        <div className="ragSegment">
          <Segmented
            ariaLabel={text("引用检查", "Citation check")}
            value={search.require_citations ? "on" : "off"}
            onChange={(value) => save("rag_search", { require_citations: value === "on" })}
            options={[{ value: "on", label: text("强制", "Enforced") }, { value: "off", label: text("关闭", "Off") }]}
          />
        </div>
        <p className="ragHint">
          {search.require_citations
            ? text("模型检索后给出答案，系统会用代码核对：引用的 [路径#片段] 必须是检索真正返回过的，引号里的原文必须在该片段里，并且必须引用（或声明“Sources: none used”）。不合格的答案会被打回重写，最多两次。", "After a search, the system checks the answer in code: every cited [path#chunk] must be one a search returned, quoted text must be in that chunk, and something must be cited (or the answer says \"Sources: none used\"). A failing answer is sent back for a rewrite, twice at most.")
            : text("不检查引用。", "Citations are not checked.")}
        </p>
      </div>
      <div className="ragActions">
        <button className="secondaryButton" onClick={reindex} disabled={loading} type="button">
          {loading ? <span className="spinner" /> : <RefreshCw size={16} />}
          <span>{loading ? text("正在刷新…", "Refreshing…") : text("刷新 RAG", "Refresh RAG")}</span>
        </button>
        <button className="secondaryButton" onClick={onUpload} disabled={uploading || loading} type="button">
          <Paperclip size={16} />
          <span>{uploading ? text("正在录入…", "Ingesting…") : text("上传文档", "Upload document")}</span>
        </button>
      </div>
      {refresh.phase !== "idle" && (
        <div className={`ragRefreshFeedback ${refresh.phase}`} role={refresh.phase === "error" ? "alert" : "status"}>
          {loading && <span className="spinner" />}
          {refresh.phase === "success" && <Check size={15} />}
          {refresh.phase === "error" && <X size={15} />}
          <span>{refresh.message}</span>
        </div>
      )}
    </section>
  );
}
