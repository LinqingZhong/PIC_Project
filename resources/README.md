# 🧠 Knowledge-Enhanced RAG Pipeline with Dual-Language Support

This repository contains a modular Retrieval-Augmented Generation (RAG) system that integrates semantic document retrieval, structured knowledge graph schema extraction, and large language model inference. It supports both Chinese and English knowledge sources and prompts.

---

## 📁 Project Structure

```
.
├── rag_retriever.py              # English knowledge retriever using all-MiniLM-L6-v2
├── rag_retriever_bert.py         # Chinese retriever based on BERT-Chinese
├── schema_retriever.py           # Graph schema retriever using all-MiniLM-L6-v2
├── rag_inference_example.py      # Full pipeline: question → schema + context → LLM output
├── resources/
│   ├── faiss_index_en/           # English FAISS index & docs.json
│   ├── faiss_index_cn/           # Chinese FAISS index & docs.json
│   ├── schema/
│   │   ├── triples_zh.index      # Chinese graph schema FAISS index
│   │   ├── triples_en.index      # English graph schema FAISS index
│   │   ├── triples_zh.pkl        # Raw Chinese triples
│   │   └── triples_en.pkl        # Raw English triples
│   ├── knowledge_en.jsonl        # English corpus
│   ├── knowledge_cn.jsonl        # Chinese corpus
│   ├── triples_en.txt            # Raw English triples for graph schema
│   └── triples_cn.txt            # Raw Chinese triples for graph schema
```

---



## 🚀 Quickstart

1. Install dependencies:
```bash
pip install faiss-cpu sentence-transformers transformers jieba
```

2. (Optional) Build FAISS indices:
```python
from rag_retriever import RAGRetriever
retriever = RAGRetriever("resources/faiss_index_en")
retriever.build_index("resources/knowledge_en.jsonl")

from rag_retriever_bert import RAGRetrieverBERT
retriever = RAGRetrieverBERT("resources/faiss_index_cn")
retriever.build_index("resources/knowledge_cn.jsonl")
```

3. Run inference:
```bash
python rag_inference_example.py
```

---

## 🧪 Example Prompt

```
🌏 中文问题：什么是新能源？
🤖 Cypher:
 You are an expert in knowledge graph and question answering.

Graph Schema:
(:地热能)-[:是]->(:来自地球深处的热能)
(:地热能不但)-[:是]->(:一种清洁能源)

...
Reference Documents:
无论是地震还是火山爆发，都是地下能量的释放过程。地球内部的总热能量约为地球上全部煤炭储量的1.7亿倍
...

🌍 English question: What is geothermal energy?
🤖 Cypher:
  You are an expert in knowledge graph and question answering.

Graph Schema:
(:Emulsions)-[:CONTAIN]->(:Salts)
...

Reference Documents:
Geothermal power generation does have some associated greenhouse gas emissions- which depend on the type of plant
...
```
---


## 📄 License
MIT
