#!/bin/sh
# Receive a Git bundle on stdin over SSH; no server-to-GitHub connection is needed.
set -eu

: "${APP_DIR:?APP_DIR is required}"
: "${REPO_URL:?REPO_URL is required}"
: "${DEPLOY_BRANCH:?DEPLOY_BRANCH is required}"
: "${DEPLOY_SHA:?DEPLOY_SHA is required}"
git check-ref-format "refs/heads/$DEPLOY_BRANCH"

bundle="$(mktemp)"
trap 'rm -f "$bundle"' EXIT
echo "Receiving deployment bundle..."
cat > "$bundle"

mkdir -p "$APP_DIR"
cd "$APP_DIR"
if [ ! -d .git ]; then
  git init -b "$DEPLOY_BRANCH"
  git remote add origin "$REPO_URL"
fi

git bundle verify "$bundle"
git fetch --no-tags "$bundle" "+HEAD:refs/remotes/origin/$DEPLOY_BRANCH"
actual_sha="$(git rev-parse FETCH_HEAD)"
if [ "$actual_sha" != "$DEPLOY_SHA" ]; then
  echo "Deployment commit mismatch: expected $DEPLOY_SHA, got $actual_sha" >&2
  exit 1
fi
git symbolic-ref HEAD "refs/heads/$DEPLOY_BRANCH"
git reset --hard "origin/$DEPLOY_BRANCH"
echo "Deploying commit $actual_sha"
SKIP_GIT_UPDATE=1 sh scripts/update_server.sh </dev/null
