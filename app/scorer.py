"""
Scoring Engine
==============
Combines computer vision similarity + Claude AI analysis
into a single final score with full breakdown.

Final Score Composition:
  50% — MobileNet visual embedding similarity
  25% — Color histogram match
  25% — Claude Vision AI score (if available)

If Claude is not configured, visual scores are reweighted:
  65% — MobileNet embedding
  35% — Color histogram
"""

from typing import Optional
from PIL import Image
from app.vision import (
    DishProfile,
    extract_features,
    compare_to_reference,
    load_image,
)
from app.claude_vision import analyze_with_claude
from app.config import settings


def analyze_food_image(
    query_image: Image.Image,
    dish_profile: DishProfile,
) -> dict:
    """
    Full analysis pipeline:
    1. Extract visual features from query image
    2. Compare against reference features (visual similarity)
    3. Send to Claude for ingredient analysis
    4. Combine into final weighted score

    Returns a complete result dict ready to send back to Laravel.
    """

    # ── Step 1: Visual feature comparison ───────────────────────────────────
    query_features = extract_features(query_image)
    reference_features = dish_profile.get_reference_features()

    if not reference_features:
        return {
            "success": False,
            "error": f"Dish '{dish_profile.dish_name}' has no reference images. Upload training images first.",
        }

    visual_score_raw, visual_breakdown = compare_to_reference(
        query_features,
        reference_features,
    )
    visual_score_pct = visual_score_raw * 100  # 0-100

    # ── Step 2: Claude ingredient analysis ──────────────────────────────────
    # Use first reference image for Claude comparison (most "canonical" one)
    reference_image = None
    if dish_profile.reference_image_paths:
        try:
            reference_image = load_image(dish_profile.reference_image_paths[0])
        except Exception:
            reference_image = None  # Reference image missing from disk, continue without it

    claude_result = analyze_with_claude(
        query_image=query_image,
        reference_image=reference_image,
        dish_name=dish_profile.dish_name,
        expected_ingredients=dish_profile.ingredients,
    )

    # ── Step 3: Combine scores ───────────────────────────────────────────────
    claude_available = claude_result["claude_score"] is not None

    if claude_available:
        # Full weighted combination
        final_score = (
            (visual_breakdown["visual_structure_score"] * settings.VISUAL_SIMILARITY_WEIGHT) +
            (visual_breakdown["color_score"] * settings.COLOR_SIMILARITY_WEIGHT) +
            (claude_result["claude_score"] * settings.CLAUDE_SCORE_WEIGHT)
        )
        score_method = "visual_embedding + color_histogram + claude_vision"
    else:
        # Reweight without Claude
        final_score = (
            (visual_breakdown["visual_structure_score"] * 0.65) +
            (visual_breakdown["color_score"] * 0.35)
        )
        score_method = "visual_embedding + color_histogram (claude unavailable)"

    final_score = round(min(max(final_score, 0.0), 100.0), 1)

    # ── Step 4: Quality label ────────────────────────────────────────────────
    quality_label = _score_to_label(final_score)

    # ── Build response ───────────────────────────────────────────────────────
    return {
        "success": True,
        "dish_id": dish_profile.dish_id,
        "dish_name": dish_profile.dish_name,

        # The main number Laravel cares about
        "match_percentage": final_score,
        "quality_label": quality_label,  # "Excellent" / "Good" / "Needs Improvement" / "Poor"

        # Ingredient-level findings from Claude
        "missing_ingredients": claude_result["missing_ingredients"],
        "issues_found": claude_result["issues_found"],
        "correct_elements": claude_result["correct_elements"],
        "overall_assessment": claude_result["overall_assessment"],

        # Score breakdown for debugging / display
        "score_breakdown": {
            "final_score": final_score,
            "visual_structure_score": visual_breakdown["visual_structure_score"],
            "color_score": visual_breakdown["color_score"],
            "claude_ai_score": claude_result["claude_score"],
            "scoring_method": score_method,
            "weights": {
                "visual_structure": f"{settings.VISUAL_SIMILARITY_WEIGHT * 100:.0f}%",
                "color": f"{settings.COLOR_SIMILARITY_WEIGHT * 100:.0f}%",
                "claude_ai": f"{settings.CLAUDE_SCORE_WEIGHT * 100:.0f}%" if claude_available else "0% (unavailable)",
            }
        },

        "reference_images_used": len(reference_features),
        "claude_confidence": claude_result["confidence"],
    }


def _score_to_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Needs Improvement"
    else:
        return "Poor"