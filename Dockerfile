FROM node:22-slim AS web
WORKDIR /src/apps/console/frontend
RUN corepack enable && corepack prepare pnpm@10.26.0 --activate
COPY apps/console/frontend/package.json apps/console/frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY apps/console/frontend ./
RUN pnpm build


FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CRYPTO_MAMBA_CORE_PYTHON=/usr/local/bin/python \
    PORT=8600

WORKDIR /app

RUN pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu

COPY apps/model-backend/requirements-inference.txt apps/model-backend/requirements-inference.txt
COPY apps/console/backend/requirements.txt apps/console/backend/requirements.txt
RUN pip install -r apps/model-backend/requirements-inference.txt -r apps/console/backend/requirements.txt

COPY apps apps
COPY evidence evidence
COPY --from=web /src/apps/console/web-dist apps/console/web-dist

# --no-deps keeps the training stack listed in model-backend/requirements.txt out of the image.
RUN pip install --no-deps -e apps/model-backend -e apps/console/backend

# The Trading screen reads the chronological backtest, which is not tracked in git.
RUN cd evidence && sha256sum -c SHA256SUMS --quiet \
 && cd ../apps/model-backend && python scripts/run_backtest.py > /dev/null

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh && useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8600
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['PORT'], timeout=4)"]
ENTRYPOINT ["entrypoint.sh"]
