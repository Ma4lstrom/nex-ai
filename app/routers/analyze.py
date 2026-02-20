"""
Analysis Router
===============
The main endpoint — send a food photo, get back a quality score.

POST /analyze/{dish_id}  → Analyze a photo against a trained dish profile
POST /analyze/batch      → Analyze multiple images at once
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from PIL import Image
import io
from app.vision import DishProfile, load_image
from app.scorer import analyze_food_image
from app.config import settings

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024


@router.post("/{dish_id}", summary="Analyze a food photo against a trained dish")
async def analyze_image(
    dish_id: str,
    image: UploadFile = File(..., description="The food photo to evaluate"),
):
    """
    Analyze a food image and return:
    - match_percentage: 0-100 overall quality score
    - quality_label: Excellent / Good / Needs Improvement / Poor
    - missing_ingredients: list of things Claude says are absent
    - issues_found: problems with presentation, color, portion, etc.
    - correct_elements: what looks right
    - score_breakdown: detailed per-component scores

    Requires the dish to have at least 1 reference image uploaded via /train/{dish_id}
    """

    # ── Load dish profile ────────────────────────────────────────────────────
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Dish '{dish_id}' not found. Create it via POST /dishes/ first.",
        )

    if not profile.reference_features:
        raise HTTPException(
            status_code=422,
            detail=f"Dish '{dish_id}' has no reference images. Upload training images via POST /train/{dish_id}",
        )

    # ── Validate uploaded image ──────────────────────────────────────────────
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{image.content_type}'. Use JPEG, PNG, or WebP.",
        )

    raw = await image.read()
    if len(raw) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(raw)/1024/1024:.1f}MB). Max is {settings.MAX_IMAGE_SIZE_MB}MB.",
        )

    # ── Load and analyze ─────────────────────────────────────────────────────
    try:
        query_image = load_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    result = analyze_food_image(query_image, profile)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

    return result


@router.post("/batch/{dish_id}", summary="Analyze multiple food photos at once")
async def analyze_batch(
    dish_id: str,
    images: List[UploadFile] = File(..., description="Multiple food photos to evaluate"),
):
    """
    Analyze multiple images in one request.
    Returns an array of results with an index for each image.
    Useful for bulk quality checks (e.g. end-of-shift audit).
    """
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found")

    if not profile.reference_features:
        raise HTTPException(
            status_code=422,
            detail=f"Dish '{dish_id}' has no reference images.",
        )

    results = []

    for i, upload in enumerate(images):
        try:
            raw = await upload.read()
            query_image = load_image(raw)
            result = analyze_food_image(query_image, profile)
            result["image_index"] = i
            result["original_filename"] = upload.filename
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "image_index": i,
                "original_filename": upload.filename,
                "error": str(e),
            })

    # Summary stats
    successful = [r for r in results if r.get("success")]
    avg_score = (
        sum(r["match_percentage"] for r in successful) / len(successful)
        if successful else 0
    )

    return {
        "dish_id": dish_id,
        "dish_name": profile.dish_name,
        "images_analyzed": len(results),
        "successful": len(successful),
        "average_score": round(avg_score, 1),
        "results": results,
    }