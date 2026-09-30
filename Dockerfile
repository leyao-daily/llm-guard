# llm-guard runs on the Python standard library, so this image needs no pip
# install step and has no third-party packages to audit.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="llm-guard" \
      org.opencontainers.image.description="Self-hosted LLM cost attribution and budget enforcement gateway" \
      org.opencontainers.image.licenses="MIT"

# Unbuffered output so `docker logs` is useful, and no .pyc writes so the
# container filesystem can stay read-only.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLMGUARD_DB=/data/usage.db

WORKDIR /app
COPY llmguard/ ./llmguard/
COPY pyproject.toml README.md ./

# Run unprivileged, and keep the database on a volume.
RUN useradd --create-home --uid 10001 llmguard \
    && mkdir -p /data \
    && chown -R llmguard:llmguard /data /app
USER llmguard

VOLUME ["/data"]
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/-/health', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "llmguard"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8787"]
