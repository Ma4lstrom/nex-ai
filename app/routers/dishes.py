"""
Dishes Router
=============
Manage dish profiles (the "what should this food look like" definitions).

POST /dishes/          → Create a new dish profile
GET  /dishes/          → List all dishes
GET  /dishes/{id}      → Get a specific dish
DELETE /dishes/{id}    → Delete a dish and its model
"""

import os
import glob
import pickle
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.vision import DishProfile
from app.config import settings

router = APIRouter()


class CreateDishRequest(BaseModel):
    dish_id: str           # e.g. "margherita_pizza" — used as filename key
    dish_name: str         # e.g. "Margherita Pizza" — human readable
    ingredients: List[str] = []  # e.g. ["mozzarella", "tomato sauce", "basil", "olive oil"]


@router.post("/", summary="Create a new dish profile")
def create_dish(request: CreateDishRequest):
    """
    Create a dish profile before uploading training images.
    The dish_id becomes the key used in all other endpoints.
    """
    # Check if already exists
    existing = DishProfile.load(request.dish_id, settings.MODEL_DIR)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Dish '{request.dish_id}' already exists. Use PUT to update or DELETE first."
        )

    profile = DishProfile(
        dish_id=request.dish_id,
        dish_name=request.dish_name,
        ingredients=request.ingredients,
    )
    profile.save(settings.MODEL_DIR)

    return {
        "success": True,
        "message": f"Dish '{request.dish_name}' created. Now upload reference images via POST /train/{request.dish_id}",
        "dish": profile.to_dict(),
    }


@router.get("/", summary="List all dish profiles")
def list_dishes():
    """Return all registered dishes and their training status."""
    pattern = os.path.join(settings.MODEL_DIR, "*.pkl")
    dish_files = glob.glob(pattern)

    dishes = []
    for filepath in dish_files:
        try:
            with open(filepath, "rb") as f:
                profile = pickle.load(f)
            dishes.append({
                **profile.to_dict(),
                "ready_for_analysis": len(profile.reference_features) > 0,
            })
        except Exception:
            pass  # Skip corrupted files

    return {"dishes": dishes, "total": len(dishes)}


@router.get("/{dish_id}", summary="Get a dish profile")
def get_dish(dish_id: str):
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found")
    return {
        **profile.to_dict(),
        "ready_for_analysis": len(profile.reference_features) > 0,
    }


@router.put("/{dish_id}/ingredients", summary="Update expected ingredients")
def update_ingredients(dish_id: str, ingredients: List[str]):
    """Update the expected ingredient list for a dish (used by Claude analysis)."""
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found")

    profile.ingredients = ingredients
    profile.save(settings.MODEL_DIR)

    return {"success": True, "dish_id": dish_id, "ingredients": ingredients}


@router.delete("/{dish_id}", summary="Delete a dish and its model")
def delete_dish(dish_id: str):
    profile = DishProfile.load(dish_id, settings.MODEL_DIR)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dish '{dish_id}' not found")

    # Delete model file
    model_path = os.path.join(settings.MODEL_DIR, f"{dish_id}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)

    # Delete reference images
    ref_dir = os.path.join(settings.REFERENCE_IMAGE_DIR, dish_id)
    if os.path.exists(ref_dir):
        import shutil
        shutil.rmtree(ref_dir)

    return {"success": True, "message": f"Dish '{dish_id}' and all training data deleted"}