import os
import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import math


class RAGRetriever:
    def __init__(self, db_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=3):
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.index = None
        self.documents = []

        if os.path.exists(os.path.join(db_path, "index.faiss")):
            self.load()
        else:
            print(f"[!] FAISS index not found in {db_path}. Please run build_index() first.")

    def build_index(self, jsonl_path: str):
        texts = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if "text" in item:
                    texts.append(item["text"].strip())

        self.documents = texts
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        self.index = index

        os.makedirs(self.db_path, exist_ok=True)
        faiss.write_index(index, os.path.join(self.db_path, "index.faiss"))
        with open(os.path.join(self.db_path, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)

        print(f"[✓] Built FAISS index with {len(texts)} documents in {self.db_path}")

    def load(self):
        self.index = faiss.read_index(os.path.join(self.db_path, "index.faiss"))
        with open(os.path.join(self.db_path, "docs.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def retrieve(self, query: str) -> List[str]:
        if not self.index:
            raise RuntimeError("FAISS index is not loaded. Please run build_index() first.")
        embedding = self.model.encode([query], convert_to_numpy=True)

        D, I = self.index.search(embedding, self.top_k * 5)

        candidates = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                length_weight = math.log(len(doc) + 1)
                score = float(np.dot(embedding, embedding.T)[0][0])  
                final_score = score * length_weight
                candidates.append((doc, final_score))

        sorted_docs = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.top_k]
        return [doc for doc, _ in sorted_docs]


# Example usage
# from rag_retriever import RAGRetriever
# retriever = RAGRetriever(db_path="resources/faiss_index_en")
# # Uncomment the next line to build the index
# # retriever.build_index("resources/knowledge_en.jsonl")
# results = retriever.retrieve("What are the sources of renewable energy?")
# for i, text in enumerate(results):
#     print(f"[{i+1}] {text}")
