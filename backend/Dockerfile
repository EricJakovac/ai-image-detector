FROM python:3.11-slim

# Instalacija sistemskih ovisnosti
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Hugging Face zahtijeva port 7860
EXPOSE 7860

# Pokreni aplikaciju koristeći varijablu okruženja za port (default na 7860 za HF)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]