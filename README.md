---
title: AI Image Detector API
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 [AI Image Detector](https://ai-image-detector-eight.vercel.app/)

Cilj projekta je razviti sustav koji može razlikovati:
- **AI-generirane slike**
- **Stvarne fotografije**

Modeli su trenirani i evaluirani na različitim datasetovima kako bi se analizirala njihova sposobnost generalizacije na novije AI generirane slike.

---

## 💻 Tehnologije

### 🧠 Machine Learning / AI
- **PyTorch** – glavni framework za treniranje i izvođenje modela  
- **Torchvision** – dataset loaderi i transformacije slika  
- **Timm (PyTorch Image Models)** – implementacije modernih arhitektura  
- **Scikit-learn** – evaluacijske metrike (accuracy, F1, ROC AUC…)  
- **NumPy** – rad s podacima i matricama  

---

### 🏗️ Modeli i arhitekture
- **EfficientNet (CNN)**  
  - fokus na teksture i lokalne uzorke  
- **ConvNeXt (moderni CNN)**  
  - kombinira CNN i transformer ideje  
- **Vision Transformer (ViT)**  
  - fokus na globalne odnose u slici  
- **DeiT (Data-efficient Image Transformer)**  
  - optimiziran transformer za manje datasetove  

---

### 🔍 Vizualizacija modela
- **Grad-CAM**  
  - koristi se za CNN modele  
  - prikazuje koje dijelove slike model koristi za odluku  
- **Attention Maps (ViT / DeiT)**  
  - prikazuje fokus transformera  

---

### ⚙️ Backend
- **FastAPI** – REST API za inferenciju modela  
- **Uvicorn** – ASGI server  
- **Pillow (PIL)** – obrada slika  
- **Python-dotenv** – upravljanje environment varijablama  

---

### 🎨 Frontend
- **React** – korisničko sučelje  
- **Axios / Fetch API** – komunikacija s backendom  
- Prikaz:
  - rezultata klasifikacije
  - confidence score
  - vizualizacija modela  

---

### 🐳 Infrastruktura
- **Docker** – containerizacija aplikacije  
- **Docker Compose** – orkestracija servisa  
- **Hugging Face Hub** – pohrana i dohvat modela  

---

### 📊 Datasetovi
Korišteni datasetovi:
- https://www.kaggle.com/datasets/ishu0505/ai-vs-real-84k-train-data  
- https://www.kaggle.com/datasets/hiddenplant/sut-project  
- https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/viewer  

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
