
import os
import faiss
import json
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

TRIPLE_TXT_CN = "resources/triples_cn.txt"
TRIPLE_TXT_EN = "resources/triples_en.txt"
INDEX_CN = "resources/schema/triples_zh.index"
INDEX_EN = "resources/schema/triples_en.index"
STORE_CN = "resources/schema/triples_zh.pkl"
STORE_EN = "resources/schema/triples_en.pkl"

invalid_terms_zh = {
    "首先", "其次", "无论", "甚至", "这", "那", "它", "其中", "例如", "这就", "那是",
    "是", "可以", "通过", "一般", "某种", "被称为", "一种"
}
invalid_terms_en = {
    "it", "this", "that", "they", "he", "she", "we", "you", "something",
    "anything", "everything", "first", "second", "third", "also", "even",
    "there", "here", "then", "thus", "so", "such", "which", "what","Fig"
}

def is_valid_entity(ent):
    ent = ent.strip().lower()
    return ent not in invalid_terms_zh and ent not in invalid_terms_en and len(ent) > 1 and not ent.isdigit()

def parse_triples(filepath):
    triples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if ")-[:" in line:
                try:
                    h = line.split("(:")[1].split(")")[0]
                    r = line.split("[:")[1].split("]")[0]
                    t = line.split("->(:")[1].split(")")[0]
                    if is_valid_entity(h) and is_valid_entity(t):
                        sentence = f"{h} {r} {t}"
                        triples.append((line.strip(), sentence, h, r, t))
                except:
                    continue
    return triples

def encode_texts_with_keywords(texts, keywords=None):
    if keywords:
        texts = [f"{kw} {t}" for t in texts for kw in keywords]
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def build_index(triples, index_path, store_path):
    print(f"[⚙️] Building index for: {index_path}")
    sentences = [t[1] for t in triples]
    embeddings = encode_texts_with_keywords(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump(triples, f)
    print(f"[✓] Indexed {len(triples)} triples")

def load_index(index_path, store_path):
    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        triples = pickle.load(f)
    return index, triples

def extract_keywords(text):
    import jieba.analyse
    return jieba.analyse.extract_tags(text, topK=3)

def retrieve_schema(question, lang="zh", top_k=5):
    if not os.path.exists(INDEX_CN):
        zh_triples = parse_triples(TRIPLE_TXT_CN)
        build_index(zh_triples, INDEX_CN, STORE_CN)
    if not os.path.exists(INDEX_EN):
        en_triples = parse_triples(TRIPLE_TXT_EN)
        build_index(en_triples, INDEX_EN, STORE_EN)
    if lang == "zh":
        index, triples = load_index(INDEX_CN, STORE_CN)
    else:
        index, triples = load_index(INDEX_EN, STORE_EN)

    keywords = extract_keywords(question)
    triples_filtered = [t for t in triples if any(kw in t[0] for kw in keywords)]

    if not triples_filtered:
        triples_filtered = triples

    candidates = [t[1] for t in triples_filtered]
    embeddings = encode_texts_with_keywords(candidates, keywords)
    sub_index = faiss.IndexFlatL2(embeddings.shape[1])
    sub_index.add(embeddings)
    q_embedding = encode_texts_with_keywords([question], keywords)
    D, I = sub_index.search(q_embedding, top_k)
    return "\n".join(triples_filtered[i][0] for i in I[0] if i < len(triples_filtered))



# example usage
# if __name__ == "__main__":
#     q_zh = "新能源是什么"
#     q_en = "How is solar energy used in transportation?"

#     print("\n[🌏 中文 Schema]")
#     print(retrieve_schema(q_zh, lang="zh"))

#     print("\n[🌍 English Schema]")
#     print(retrieve_schema(q_en, lang="en"))
