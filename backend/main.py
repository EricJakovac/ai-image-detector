from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ NOVO
from api.v1.test_v1_endpoint import router as test_router
from api.v1.upload_predict import router as upload_router
import os

app = FastAPI(title="AI Image Detector Backend", version="0.1.0")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# ✅ CORS middleware - dozvoli React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(test_router, prefix="/api/v1", tags=["test"])
app.include_router(upload_router, prefix="/api/v1", tags=["predict"])


@app.get("/")
def read_root():
    return {"status": "ok", "message": "backend + CORS ready"}
