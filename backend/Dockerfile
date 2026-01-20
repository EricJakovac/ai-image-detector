FROM python:3.11-slim
# Umjesto FROM python:3.11 
# Instaliraj sistemske ovisnosti za OpenCV jer 
# standardna verzija python:3.11 je puno veca nego -slim sa dodatnim paketima
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .