"""
Training Router
===============
Upload reference images for a dish (the "correct" examples).

POST /train/{dish_id}         → Upload one or more reference images
DELETE /train/{dish_id}/reset → Remove all reference images, keep dish profile
"""

import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from PIL import Image
import io
from app.vision import DishProfile, load_image
from app.config import settings

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024


@router.post("/{dish_id}", summary="Upload reference images for a dish")
async def upload_reference_images(
    dish_id: str,
    images: List[UploadFile] = File(..., description="One or more reference food images"),
):
    """
    Upload reference images that define how this dish SHOULD look.

    - More images = better accuracy (recommended: 3-10 from different angles)
    - Images are stored and feature vectors extracted immediately
    - Re-uploading adds to existing references (doesn't replace them)
    """
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Dish '{dish_id}' not found. Create it first via POST /dishes/",
        )

    # Ensure reference storage dir for this dish
    dish_ref_dir = os.path.join(settings.REFERENCE_IMAGE_DIR, dish_id)
    os.makedirs(dish_ref_dir, exist_ok=True)

    processed = []
    errors = []

    for upload in images:
        try:
            # Validate type
            if upload.content_type not in ALLOWED_TYPES:
                errors.append({
                    "filename": upload.filename,
                    "error": f"Unsupported type '{upload.content_type}'. Use JPEG, PNG, or WebP.",
                })
                continue

            # Read and validate size
            raw = await upload.read()
            if len(raw) > MAX_SIZE_BYTES:
                errors.append({
                    "filename": upload.filename,
                    "error": f"File too large ({len(raw) / 1024 / 1024:.1f}MB). Max is {settings.MAX_IMAGE_SIZE_MB}MB.",
                })
                continue

            # Load image
            image = load_image(raw)

            # Save to disk with unique name
            ext = "jpg" if upload.content_type == "image/jpeg" else upload.content_type.split("/")[1]
            filename = f"{uuid.uuid4().hex}.{ext}"
            save_path = os.path.join(dish_ref_dir, filename)
            image.save(save_path, quality=90)

            # Extract features and add to profile
            profile.add_reference(image, save_path)

            processed.append({
                "filename": upload.filename,
                "saved_as": filename,
                "size": f"{image.size[0]}x{image.size[1]}",
            })

        except Exception as e:
            errors.append({"filename": upload.filename, "error": str(e)})

    # Save updated profile
    if processed:
        profile.save(settings.MODEL_DIR)

    return {
        "success": len(processed) > 0,
        "dish_id": dish_id,
        "dish_name": profile.dish_name,
        "images_added": len(processed),
        "total_references": len(profile.reference_features),
        "processed": processed,
        "errors": errors,
        "ready_for_analysis": len(profile.reference_features) > 0,
    }


@router.delete("/{dish_id}/reset", summary="Reset all reference images for a dish")
def reset_references(dish_id: str):
    """Remove all reference images and retrain from scratch."""
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found")

    # Clear features
    profile.reference_features = []
    profile.reference_image_paths = []

    # Delete stored images
    dish_ref_dir = os.path.join(settings.REFERENCE_IMAGE_DIR, dish_id)
    if os.path.exists(dish_ref_dir):
        shutil.rmtree(dish_ref_dir)
        os.makedirs(dish_ref_dir)

    profile.save(settings.MODEL_DIR)

    return {
        "success": True,
        "message": f"All reference images for '{dish_id}' removed. Upload new ones to retrain.",
    }