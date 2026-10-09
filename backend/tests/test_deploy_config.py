from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_compose_keeps_agent_internal_on_shared_lease_network():
    compose = read("docker-compose.yml")

    assert "ai-agent-backend:" in compose
    assert "ai-agent-frontend:" in compose
    assert "ports:" not in compose
    assert "lease-ai-net:" in compose
    assert "external: true" in compose
    assert "LEASE_MCP_URL: ${LEASE_MCP_URL:-http://web:5000/mcp}" in compose
    assert "VITE_API_BASE: /ai-agent" in compose


def test_backend_entrypoint_defaults_to_lease_mcp():
    entrypoint = read("docker/backend-entrypoint.sh")

    assert '"url": os.environ.get("LEASE_MCP_URL", "http://web:5000/mcp")' in entrypoint
    assert "servers.local.json" in entrypoint


def test_backend_dockerfile_caches_dependency_install_layer():
    dockerfile = read("docker/backend.Dockerfile")

    pyproject_copy = dockerfile.index("COPY backend/pyproject.toml")
    dependency_install = dockerfile.index("pip', 'install'")
    backend_copy = dockerfile.index("COPY backend ./backend")

    assert pyproject_copy < dependency_install < backend_copy


def test_frontend_dockerfile_reuses_npm_downloads_and_retries_network():
    dockerfile = read("docker/frontend.Dockerfile")

    package_copy = dockerfile.index("COPY frontend/package*.json")
    dependency_install = dockerfile.index("npm ci")
    frontend_copy = dockerfile.index("COPY frontend ./")

    assert package_copy < dependency_install < frontend_copy
    assert "--mount=type=cache,target=/root/.npm" in dockerfile
    assert '--registry="${NPM_REGISTRY}"' in dockerfile
    assert "--fetch-retries=2" in dockerfile
    assert "npm run build -- --base=/ai-agent/" in dockerfile


def test_deploy_creates_shared_network_and_checks_gateway():
    update_script = read("scripts/update_server.sh")
    deploy_workflow = read(".github/workflows/deploy.yml")

    assert 'docker network create "$SHARED_NETWORK"' in update_script
    assert "sh scripts/migrate_from_lease.sh" in update_script
    assert 'docker compose -f "$COMPOSE_FILE" up -d --build' in update_script
    assert "Deploy update complete." in update_script
    assert "git pull --ff-only && exec sh scripts/update_server.sh" in deploy_workflow
    assert "ai via gateway ready" in deploy_workflow
