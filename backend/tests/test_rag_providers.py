"""The HTTP layer of `/embeddings` and `/rerank`, against a real local server (no mocks of httpx)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from agent.config import ModelConfig
from tools.rag import api_embeddings, providers, rerank
from tools.rag.models import ScoredChunk


class Provider:
    def __init__(self):
        self.requests = []
        self.failures = []  # status codes to answer with before succeeding
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append((self.path, self.headers.get("Authorization"), body))
                if provider.failures:
                    status = provider.failures.pop(0)
                    self.send_response(status)
                    self.end_headers()
                    self.wfile.write(b"overloaded")
                    return
                if self.path.endswith("/embeddings"):
                    # the provider may answer out of order: `index` is what counts
                    data = [{"index": i, "embedding": [float(i + 1), 0.0, 1.0]} for i in range(len(body["input"]))][::-1]
                    payload = {"data": data}
                else:
                    payload = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def provider(monkeypatch):
    server = Provider()
    config = ModelConfig(provider="local", model_id="embed-1", base_url=server.base_url, api_key="secret-key")
    monkeypatch.setattr(providers, "get_model_config", lambda model: config)
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    yield server
    server.close()


def test_embeddings_are_requested_with_the_providers_model_and_key_and_ordered_by_index(provider):
    matrix = api_embeddings.embed_texts("local:embed-1", ["a", "b", "c"])

    path, authorization, body = provider.requests[0]
    assert path == "/v1/embeddings" and authorization == "Bearer secret-key"
    assert body == {"input": ["a", "b", "c"], "model": "embed-1"}
    assert matrix.shape == (3, 3)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)  # unit length, so a dot product is the cosine
    assert matrix[0][0] < matrix[2][0]  # row 0 is the vector the provider labelled index 0, not the first it sent back


def test_texts_are_sent_in_batches(provider, monkeypatch):
    monkeypatch.setattr(api_embeddings, "BATCH_SIZE", 2)

    matrix = api_embeddings.embed_texts("local:embed-1", ["a", "b", "c", "d", "e"])

    assert [len(body["input"]) for _path, _auth, body in provider.requests] == [2, 2, 1] and len(matrix) == 5


def test_overloaded_responses_are_retried_but_client_errors_are_not(provider):
    provider.failures = [503, 429]

    assert len(api_embeddings.embed_texts("local:embed-1", ["a"])) == 1
    assert len(provider.requests) == 3

    provider.requests.clear()
    provider.failures = [401]
    with pytest.raises(RuntimeError, match="HTTP 401"):
        api_embeddings.embed_texts("local:embed-1", ["a"])
    assert len(provider.requests) == 1


def test_a_provider_that_keeps_failing_gives_up_with_its_message(provider):
    provider.failures = [500, 500, 500]

    with pytest.raises(RuntimeError, match="HTTP 500: overloaded"):
        api_embeddings.embed_texts("local:embed-1", ["a"])


def test_a_missing_key_is_reported_before_any_request(monkeypatch):
    config = ModelConfig(provider="openai", model_id="m", base_url="http://127.0.0.1:9/v1", api_key_env="NOT_SET_ANYWHERE")
    monkeypatch.setattr(providers, "get_model_config", lambda model: config)
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)

    with pytest.raises(ValueError, match="Missing API key for provider openai"):
        providers.post_json("openai:m", "/embeddings", {"input": ["a"]})


def test_the_rerank_endpoint_round_trip(provider):
    candidates = [ScoredChunk("knowledge", "a.md", 0, 0.5, "alpha"), ScoredChunk("knowledge", "b.md", 0, 0.4, "beta")]

    result = rerank.rerank("which is beta", candidates, mode="api", model="local:embed-1")

    path, _auth, body = provider.requests[0]
    assert path == "/v1/rerank" and body["query"] == "which is beta" and body["top_n"] == 2
    assert [chunk.path for chunk in result] == ["b.md", "a.md"] and result[0].score == 0.9


def test_cached_vectors_are_reused_and_dropped_with_their_chunks(provider, rag_env):
    settings = rag_env.settings()
    api_embeddings.embed_with_cache("local:embed-1", ["a", "b"], settings, force=False)
    provider.requests.clear()

    _matrix, report = api_embeddings.embed_with_cache("local:embed-1", ["b", "c"], settings, force=False)

    assert report == {"embedded_texts": 1, "embedding_reused": 1} and provider.requests[0][2]["input"] == ["c"]
    assert set(api_embeddings.load_cache(settings)) == {api_embeddings.text_key("local:embed-1", t) for t in ("b", "c")}
