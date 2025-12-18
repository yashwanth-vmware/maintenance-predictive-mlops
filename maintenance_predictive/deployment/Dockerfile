# Hugging Face Space - Docker runtime
# This container serves a Streamlit app on port 7860

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy app code
COPY app.py /app/app.py

# Hugging Face Spaces uses 7860 by default
EXPOSE 7860

# Streamlit must listen on 0.0.0.0:7860
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]
