from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ NOVO
from app.api.v1.test_v1_endpoint import router as test_router
from app.api.v1.upload_predict import router as upload_router

app = FastAPI(title="AI Image Detector Backend", version="0.1.0")

# ✅ CORS middleware - dozvoli React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(test_router, prefix="/api/v1", tags=["test"])
app.include_router(upload_router, prefix="/api/v1", tags=["predict"])

@app.get("/")
def read_root():
    return {"status": "ok", "message": "backend + CORS ready"}
