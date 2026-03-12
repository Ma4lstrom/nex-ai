from simplevecdb import VectorDB
import uuid

db = VectorDB("mistakes.db")
collection = db.collection("imgs")


def store_image_embedding(dish_name: str, img_emb: list[float], issue_txt: str):
    # Validate embedding dimensions
    if len(img_emb) != 1280:
        print(f"Invalid embedding dimensions: {len(img_emb)}, expected 1280")
        # Option A: Reject the embedding
        return {"success": False, "error": "Invalid embedding dimensions"}
        
        # Option B: Pad/Truncate to 1280
        if len(img_emb) < 1280:
            img_emb = img_emb + [0.0] * (1280 - len(img_emb))
        else:
            img_emb = img_emb[:1280]
    
    collection.add_texts(
        texts=[issue_txt],
        embeddings=[img_emb],
        metadatas=[{
            "issue": issue_txt,
            "referenced_dish": dish_name,
            "embeddings": img_emb,
            "dimensions": len(img_emb)
        }]
    )
    return "Done"