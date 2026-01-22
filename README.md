# 🤖 AI Image Detector

Ovaj projekt koristi duboko učenje (Deep Learning) za detekciju slika generiranih umjetnom inteligencijom. 

## 💻 Tehnologije

### 1. Backend (FastAPI - Python)
- **Arhitekture modela:**
    - **EfficientNet (CNN):** Fokusira se na teksture i lokalne anomalije u pikselima.
    - **Vision Transformer (ViT):** Fokusira se na globalne relacije i strukturu slike.
- **Vizualizacija:**
    - **Grad-CAM** za CNN (prikazuje gdje model "gleda").
    - **Attention Maps** za ViT (prikazuje fokus transformera).
- **Hugging Face Hub:** Automatsko preuzimanje naučenih utega modela (weights) prilikom prvog pokretanja.

### 2. Frontend (React)
- Moderan UI za upload slika i prikaz paralelnih rezultata iz oba modela.
- Prikaz vizualizacijskih mapa direktno na sučelju.

### 3. Infrastruktura (Docker & Docker Compose)
- Omogućuje da aplikacija radi identično na svakom računalu bez ručne instalacije biblioteka.

---

## 🚀 Pokretanje aplikacije

1. **Kloniraj projekt** ili napravi `pull` s GitHuba.

2. **Kreiraj `.env` datoteke** u frontend i backend mapama:
> ⚠️ **Napomena:** `.env` datoteke sadrže osjetljive podatke i već su dodane u `.gitignore`.

- **frontend/.env**
     ```env
     REACT_APP_API_URL=http://localhost:8000
     ```
- **backend/.env**
     ```env
    FRONTEND_URL=http://localhost:3000
    **HF_TOKEN=pogledaj poruku na wappu**
    REPO_ID=EricJakovac/ai-image-detector-model
    PYTHONUNBUFFERED=1
    OMP_NUM_THREADS=1
     ```

3. Pokretanje sustava - naredba u terminalu
```bash
docker-compose up --build
```

4. Nakon sto se servisi pokrenu, provjeri u terminalu jesu li se ispisale poruke "📥 Model CNN/VIT nije pronađen lokalno, Skidam sa Hugging Face..." te postoje li podfolderi u backend/models:
- cnn_efficientnet/model.pth
- vit_transformer/model.pth

Ako modeli postoje, aplikacija će raditi ispravno. Ako ne, ugasi servise naredbom:
```bash
docker-compose stop
``` 

Te ih ponovno pokreni naredbom: 

```bash
docker-compose up
```

### Pristup aplikaciji:

| Servis | URL | Opis |
| :--- | :--- | :--- |
| **Frontend** | `http://localhost:3000` | Korisničko sučelje (React) |
| **Backend** | `http://localhost:8000` | API (FastAPI) |
| **Dokumentacija** | `http://localhost:8000/docs` | Swagger UI za testiranje API-ja |


### Tehnički sažetak

Ključnih stavki:
- PyTorch: Glavni framework za pokretanje AI modela.
- Uvicorn: ASGI server koji pokreće FastAPI.
- Pillow (PIL): Za obradu i transformaciju slika prije predikcije.
- Timm (PyTorch Image Models): Biblioteka iz koje su povučene bazne arhitekture modela.
- DirectML (opcionalno lokalno): Korišteno tijekom treniranja za ubrzanje na Windows/AMD hardveru, dok Docker verzija koristi CPU radi maksimalne kompatibilnosti.