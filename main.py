from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import cohere

DATA_DIR = "./data"
COLLECTION = "argo_floats"
VECTOR_SIZE = 768

app = FastAPI(title="ARGO Unified API")

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
co = cohere.Client(os.getenv("COHERE_API_KEY", ""))

qdrant = QdrantClient(":memory:")
qdrant.recreate_collection(
    COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

documents = []

def index_data():
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    for file in data_files:
        float_id = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        for i, row in df.iterrows():
            text = " | ".join(f"{c}: {row[c]}" for c in df.columns)
            documents.append({"float_id": float_id, "text": text})
    
    vectors = embedder.encode([d["text"] for d in documents]).tolist()
    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload=documents[i]
        )
        for i in range(len(documents))
    ]
    qdrant.upsert(COLLECTION, points)

index_data()

class Query(BaseModel):
    query: str
    floatId: Optional[str] = None

@app.get("/")
def root():
    return {"status": "Backend is deployed & live!"}

@app.get("/floats")
def list_floats():
    ids = sorted({d["float_id"] for d in documents})
    return {"floats": ids}

@app.get("/float/{floatId}/data")
def get_float_data(floatId: str):
    file_path = f"{DATA_DIR}/{floatId}.csv"
    if not os.path.exists(file_path):
        raise HTTPException(404, "Float not found")
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")

@app.post("/ask")
def ask(q: Query):
    vector = embedder.encode([q.query]).tolist()[0]

    filter_query = {"must": [{"key": "float_id", "match": {"value": q.floatId}}]} if q.floatId else None

    results = qdrant.search(
        COLLECTION,
        query_vector=vector,
        filter=filter_query,
        limit=6
    )
    context = "\n".join(r.payload["text"] for r in results)

    response = co.chat(
        model="command",
        message=f"Context:\n{context}\n\nAnswer politely: {q.query}"
    )

    return {"answer": response.text, "context": context}
