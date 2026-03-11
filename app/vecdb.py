from simplevecdb import VectorDB
import uuid

db = VectorDB("mistakes.db")
collection = db.collection("imgs")


def store_image_embedding(dish_name: str, img_emb: list[float], issue_txt: str):
    collection.add_texts(
        texts=[issue_txt],
        embeddings=[img_emb],
        metadatas=[{
            "issue": issue_txt,
            "referenced_dish": dish_name,
            "embeddings": img_emb
        }]
    )


    return "Done"