"""
Computer Vision Engine
"""

import os
import pickle
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import io

# Load MobileNetV2 once at import time
print("🧠 Loading MobileNetV2...")
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _preprocess_fn
from tensorflow.keras.models import Model as _KerasModel

_base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
_tf_model = _KerasModel(inputs=_base.input, outputs=_base.output)
_preprocess = _preprocess_fn
print("✅ MobileNetV2 ready")


def _get_model():
    return _tf_model, _preprocess


def load_image(source) -> Image.Image:
    if isinstance(source, (str, os.PathLike)):
        return Image.open(source).convert("RGB")
    elif isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    elif isinstance(source, Image.Image):
        return source.convert("RGB")
    raise ValueError(f"Unsupported image source type: {type(source)}")


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def extract_embedding(image: Image.Image) -> np.ndarray:
    model, preprocess = _get_model()
    img = image.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess(arr)
    embedding = model.predict(arr, verbose=0)[0]
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-8)


def extract_color_histogram(image: Image.Image, bins: int = 32) -> np.ndarray:
    hist_r = np.histogram(np.array(image)[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(np.array(image)[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(np.array(image)[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / (norm + 1e-8)


def extract_features(image: Image.Image) -> dict:
    return {
        "embedding": extract_embedding(image),
        "color_histogram": extract_color_histogram(image),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def compare_to_reference(
    query_features: dict,
    reference_features_list: List[dict],
    embedding_weight: float = 0.65,
    color_weight: float = 0.35,
) -> Tuple[float, dict]:
    best_score = 0.0
    best_breakdown = {}

    for ref_features in reference_features_list:
        emb_sim = cosine_similarity(query_features["embedding"], ref_features["embedding"])
        color_sim = cosine_similarity(query_features["color_histogram"], ref_features["color_histogram"])
        combined = (emb_sim * embedding_weight) + (color_sim * color_weight)

        if combined > best_score:
            best_score = combined
            best_breakdown = {
                "visual_structure_score": round(emb_sim * 100, 1),
                "color_score": round(color_sim * 100, 1),
                "combined_visual_score": round(combined * 100, 1),
            }

    return best_score, best_breakdown


class DishProfile:
    def __init__(self, dish_id: str, dish_name: str, ingredients: List[str] = None):
        self.dish_id = dish_id
        self.dish_name = dish_name
        self.ingredients = ingredients or []
        self.reference_features: List[dict] = []
        self.reference_image_paths: List[str] = []

    def add_reference(self, image: Image.Image, image_path: str):
        features = extract_features(image)
        self.reference_features.append({
            "embedding": features["embedding"].tolist(),
            "color_histogram": features["color_histogram"].tolist(),
            "source_path": image_path,
        })
        self.reference_image_paths.append(image_path)

    def get_reference_features(self) -> List[dict]:
        return [
            {
                "embedding": np.array(f["embedding"]),
                "color_histogram": np.array(f["color_histogram"]),
            }
            for f in self.reference_features
        ]

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{self.dish_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(dish_id: str, model_dir: str) -> Optional["DishProfile"]:
        path = os.path.join(model_dir, f"{dish_id}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def to_dict(self) -> dict:
        return {
            "dish_id": self.dish_id,
            "dish_name": self.dish_name,
            "ingredients": self.ingredients,
            "reference_count": len(self.reference_features),
            "reference_paths": self.reference_image_paths,
        }
