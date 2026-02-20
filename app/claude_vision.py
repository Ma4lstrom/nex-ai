"""
AI Vision Analysis using OpenAI GPT-4o
"""

import base64
import json
import re
from PIL import Image
from typing import Optional
from app.config import settings
from app.vision import image_to_bytes


def encode_image_base64(image: Image.Image) -> str:
    img_bytes = image_to_bytes(image.resize((800, 800), Image.LANCZOS))
    return base64.standard_b64encode(img_bytes).decode("utf-8")


def analyze_with_claude(
    query_image: Image.Image,
    reference_image: Optional[Image.Image],
    dish_name: str,
    expected_ingredients: list[str],
) -> dict:

    if not settings.ANTHROPIC_API_KEY:
        return _fallback_response("No OPENAI_API_KEY set in .env")

    from openai import OpenAI
    client = OpenAI(api_key=settings.ANTHROPIC_API_KEY)

    messages_content = []

    if reference_image:
        messages_content.append({
            "type": "text",
            "text": "**REFERENCE IMAGE** (how the dish should look):"
        })
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_base64(reference_image)}"
            }
        })

    messages_content.append({
        "type": "text",
        "text": "**IMAGE TO EVALUATE** (what was actually prepared):"
    })
    messages_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image_base64(query_image)}"
        }
    })

    ingredients_str = ", ".join(expected_ingredients) if expected_ingredients else "not specified"

    messages_content.append({
        "type": "text",
        "text": f"""You are a professional food quality inspector analyzing a prepared dish.

Dish name: {dish_name}
Expected ingredients/components: {ingredients_str}

{"Compare the EVALUATION IMAGE against the REFERENCE IMAGE above." if reference_image else "Analyze the evaluation image against the expected ingredients listed."}

Respond ONLY with a raw JSON object, no markdown, no backticks:

{{
  "claude_score": <integer 0-100>,
  "missing_ingredients": [<list of absent ingredients>],
  "issues_found": [<list of problems with presentation, color, portion>],
  "correct_elements": [<list of things that look right>],
  "overall_assessment": "<one sentence summary>",
  "confidence": "<high|medium|low>"
}}"""
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": messages_content}]
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        return {
            "claude_score": int(result.get("claude_score", 0)),
            "missing_ingredients": result.get("missing_ingredients", []),
            "issues_found": result.get("issues_found", []),
            "correct_elements": result.get("correct_elements", []),
            "overall_assessment": result.get("overall_assessment", ""),
            "confidence": result.get("confidence", "medium"),
            "analysis_source": "gpt4o_vision",
        }

    except json.JSONDecodeError as e:
        return _fallback_response(f"GPT returned non-JSON response: {e}")
    except Exception as e:
        return _fallback_response(f"OpenAI API error: {e}")


def _fallback_response(reason: str) -> dict:
    return {
        "claude_score": None,
        "missing_ingredients": [],
        "issues_found": [],
        "correct_elements": [],
        "overall_assessment": f"AI ingredient analysis unavailable: {reason}",
        "confidence": "none",
        "analysis_source": "unavailable",
    }
