from simplevecdb import VectorDB
import uuid

db = VectorDB("mistakes.db")
collection = db.collection("imgs")


def store_image_embedding(dish_name: str, img_emb: list[float], issue_txt: str):
    collection.add(
        id=[f"plate_{uuid.uuid4()}"],
        embeddings=[img_emb],
        metadata=[{
            "issue": issue_txt,
            "referenced_dish": dish_name,
            "embedding": img_emb
        }]
    )

    return "Done"