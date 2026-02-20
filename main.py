"""
Food Vision API - Main Entry Point
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import dishes, analyze, training
from app.config import settings
import os

app = FastAPI(
    title="Food Vision API",
    description="AI-powered API for food image analysis and dish recognition",
    version="1.0.0",
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