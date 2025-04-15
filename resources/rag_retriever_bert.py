
import os
import json
import faiss
import numpy as np
import torch
from typing import List
from transformers import BertTokenizer, BertModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class BertSentenceEmbedder:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_embeddings = outputs.last_hidden_state * attention_mask
                summed = masked_embeddings.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                mean_embeddings = summed / counts  # mean pooling
                embeddings.append(mean_embeddings)
        return torch.cat(embeddings, dim=0).cpu().numpy()

class RAGRetrieverBERT:
    def __init__(self, db_path="faiss_index_zh_bert", top_k=5):
        self.top_k = top_k
        self.db_path = db_path
        self.index = None
        self.documents = []
        self.encoder = BertSentenceEmbedder()

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
        embeddings = self.encoder.encode(texts)

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
        embedding = self.encoder.encode([query])
        D, I = self.index.search(embedding, self.top_k * 5)  
        candidates = []
        for idx in I[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = np.dot(embedding, embedding.T)[0][0]
                weight = np.log(len(doc) + 1)
                candidates.append((doc, score * weight))
        sorted_docs = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.top_k]
        return [doc for doc, _ in sorted_docs]


# Example usage

# from rag_retriever_bert import RAGRetrieverBERT
# retriever = RAGRetrieverBERT(db_path="resources/faiss_index_cn")

# # Uncomment the following line to build the index
# # retriever.build_index("resources/knowledge_cn_cleaned_filtered_20.jsonl")
# results = retriever.retrieve("能源是什么")
# for i, r in enumerate(results):
#     print(f"[{i+1}] {r}")