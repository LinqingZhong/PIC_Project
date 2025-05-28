from rag_retriever import RAGRetriever
from rag_retriever_bert import RAGRetrieverBERT

def is_chinese(text):
    zh_count = sum('\u4e00' <= ch <= '\u9fff' for ch in text)
    ratio = zh_count / max(len(text), 1)
    return ratio > 0.3  

def detect_lang(text):
    return "zh" if is_chinese(text) else "en"

retriever_zh = RAGRetrieverBERT(db_path="/mnt/data4/zlq/PIC_Project/resources/faiss_index_cn")
retriever_en = RAGRetriever(db_path="/mnt/data4/zlq/PIC_Project/resources/faiss_index_en")
retriever_zh.load()
retriever_en.load()


def get_rag_context(question):
    lang = detect_lang(question)
    print(f"Detected language: {lang}")
    if lang == "zh-cn" or lang == "zh":
        docs = retriever_zh.retrieve(question)
        return "\n".join(docs), lang
    else:
        docs = retriever_en.retrieve(question)
        return "\n".join(docs), lang

# def rag_generate(model_name_or_path, question):
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
#     model.eval()
#     context, lang = get_rag_context(question)
    # schema = retrieve_schema(question, lang=lang)
    # prompt = f"""You are an expert in knowledge graph and question answering.

    #     Graph Schema:
    #     {schema}

    #     Reference Documents:
    #     {context}

    #     Question: {question}

    #     Answer (Cypher):
    #     """
    # return prompt
    # prompt = context
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=256,
    #         do_sample=True,
    #         temperature=0.7,
    #         top_p=0.9
    #     )
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return response[len(prompt):].strip()


# question_zh = "什么是地热能？"
# question_en = "How to process crude oil ?"

# print("🌏 中文问题：", question_zh)
# print("🤖 Cypher:\n", rag_generate(question_zh))

# print("\n🌍 English question:", question_en)
# print("🤖 Cypher:\n", rag_generate(question_en))
