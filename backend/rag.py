from dotenv import load_dotenv
load_dotenv()

import os
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

# --- Load once (important) ---
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    os.getenv("DB_NAME")
)

MODEL_NAME = os.getenv("MODEL_NAME")


# --- Retrieval ---
def retrieve_context(query, n_results=3):

    embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results
    )

    return "\n\n".join(results["documents"][0])


# --- Generation ---
def generate_answer(query, context):

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use only the provided context. "
                    "If unsure, say you don't know."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        stream=False
    )

    return response["message"]["content"]


# --- Single entrypoint (IMPORTANT) ---
def ask_question(query):

    context = retrieve_context(query)
    answer = generate_answer(query, context)

    return answer