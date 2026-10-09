# syntax=docker/dockerfile:1

FROM node:22-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./
ARG NPM_REGISTRY=https://registry.npmmirror.com
RUN --mount=type=cache,target=/root/.npm \
    npm ci \
      --registry="${NPM_REGISTRY}" \
      --replace-registry-host=always \
      --prefer-offline \
      --no-audit \
      --no-fund \
      --fetch-retries=2 \
      --fetch-retry-factor=2 \
      --fetch-retry-mintimeout=5000 \
      --fetch-retry-maxtimeout=30000 \
      --fetch-timeout=60000

COPY frontend ./

ARG VITE_API_BASE=/ai-agent
ENV VITE_API_BASE=${VITE_API_BASE}

RUN npm run build -- --base=/ai-agent/

FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/frontend.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
