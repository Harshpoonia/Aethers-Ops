FROM python:3.11-slim
WORKDIR /app
# CPU-only PyTorch (much smaller than the default CUDA build; fits small hosts)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt
COPY . .
ENV PORT=8000
CMD ["sh","-c","uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
