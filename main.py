"""
Food Vision API - Main Entry Point
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routers import dishes, analyze, training
from app.config import settings
import os
import secrets


async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")

    # constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")

app = FastAPI(
    title="Food Vision API",
    description="AI-powered API for food image analysis and dish recognition",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dishes.router, prefix="/dishes", tags=["Dishes"])
app.include_router(training.router, prefix="/training", tags=["Training"])
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])


@app.get("/")
def root():
    return {
        "Service": "Food Vision API",
        "status": "running",
        "docs": "/docs",
    }


@app.on_event("startup")
async def startup_event():
    os.makedirs(settings.REFERENCE_IMAGE_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_IMAGE_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    print("Food vision api started on port 8000")