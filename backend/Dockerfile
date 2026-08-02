FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
# Pre-install the CPU-only torch build before resolving the rest of the
# dependencies: the default PyPI wheel bundles CUDA runtime libraries that
# are both unused (no GPU on this deploy target) and a significant chunk of
# this process's memory/disk footprint. sentence-transformers' own torch
# dependency is then already satisfied and won't pull in the GPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
