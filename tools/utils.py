import json
import clip
import sys
sys.path.append("/mnt/data4/zlq/PIC_Project/resources")
import torch
import random
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from rag_inference_example import get_rag_context


clip_model, _ = clip.load("/mnt/data4/zlq/pkgs/ViT-B-32.pt", device="cuda", jit=False)

def sampling(json_path, num):
    with open(json_path, 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    with open(json_path.replace(".json", "_val.json"), 'w') as f:
        json.dump(data[:num], f, indent = 4)


def compute_clip_score(text1, text2):
    with torch.no_grad():
        text_tokens = clip.tokenize([text1, text2]).to("cuda")
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = torch.nn.functional.cosine_similarity(
            text_features[0].unsqueeze(0),
            text_features[1].unsqueeze(0)
        ).item()

    return round(similarity, 4)


def compute_nlp_metrics(reference, candidate):
    ref_tokens = word_tokenize(reference)
    cand_tokens = word_tokenize(candidate)

    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

    meteor = single_meteor_score(ref_tokens, cand_tokens)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougeL = scores['rougeL'].fmeasure

    return {
        "bleu": round(bleu, 4),
        "meteor": round(meteor, 4),
        "rouge1": round(rouge1, 4),
        "rouge2": round(rouge2, 4),
        "rougeL": round(rougeL, 4)
    }


def evaluate_and_save(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = []

    for example in tqdm(data, desc="Evaluating"):
        reference = example["answer"]
        models = [k for k in example.keys() if k not in ["question", "answer"]]

        for model_key in models:
            response = example[model_key]
            # clip_score = compute_clip_score(reference, response)
            nlp_score = compute_nlp_metrics(reference, response)

            # example[f"{model_key}_clip"] = clip_score
            example[f"{model_key}_nlp"] = nlp_score

        updated_data.append(example)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)


def get_lora_model(model_name_or_path, lora_path, output_path):
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",torch_dtype=torch.bfloat16)
 
    lora_model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16)

    model = lora_model.merge_and_unload()
 
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)



def chat_with_merged_model(model_path, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenized_chat = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def model_generation(model_path, data_json_path, key_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    with open(data_json_path, 'r') as f:
        data = json.load(f)

    for idx, item in enumerate(tqdm(data, desc="Processing", unit="item")):
        question = item["question"]
        messages = [
            {"role": "system", "content": "You are ChatBot developped by six students from ECPK."},
            {"role": "user", "content": "You are an expert in the energy field and are familiar with various petrochemical knowledge. You need to answer the following question: " + question},
        ]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # breakpoint()
        data[idx][key_name] = response
    
    with open(data_json_path.replace(".json", "_"+key_name+".json"), 'w') as f:
        json.dump(data, f, indent = 4)

def model_generation_w_rag(model_path, data_json_path, key_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    with open(data_json_path, 'r') as f:
        data = json.load(f)

    for idx, item in enumerate(tqdm(data, desc="Processing", unit="item")):
        question = item["question"]
        refs = get_rag_context(question)
        messages = [
            {"role": "system", "content": "You are ChatBot developped by six students from ECPK."},
            {"role": "user", "content": f"You are an expert in the energy field and are familiar with various petrochemical knowledge. Here are some references: {refs}\n\nYou need to answer the following question directly: " + question},
        ]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
        ]
        breakpoint()
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        data[idx][key_name] = response
    
    with open(data_json_path.replace(".json", "_"+key_name+".json"), 'w') as f:
        json.dump(data, f, indent = 4)

if __name__ == "__main__":
    base_model_path = "/mnt/data4/zlq/pkgs/Qwen2.5-7B"
    # lora_checkpoint_path = "/home/zlq/PIC/exps/qwen2.5_sft/checkpoint-897"
    # output_path = "/home/zlq/PIC/qwen"
    # get_lora_model(model_name_or_path=base_model_path, lora_path=lora_checkpoint_path, output_path=output_path)
    # print(chat_with_merged_model("/mnt/data4/zlq/pkgs/llama-3.1-8B", "Who are you?"))
    # model_generation("/home/zlq/PIC/llama", "/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val.json", "sft_llama")
    # model_generation("/home/zlq/PIC/internlm3", "/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val.json", "sft_internlm3")
    # model_generation("/home/zlq/PIC/qwen", "/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val.json", "sft_qwen")
    
    # path1 = "/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val_sft_internlm3.json"
    # path2 = "/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val_sft_qwen.json"
    # path3 = "/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val.json"
    # with open(path1, 'r') as f:
    #     a = json.load(f)
    # with open(path2, 'r') as f:
    #     b = json.load(f)
    # with open(path3, 'r') as f:
    #     c = json.load(f)
    
    # for i, item in enumerate(a):
    #     c[i]["sft_internlm3"] = a[i]["sft_internlm3"]
        # c[i]["sft_qwen"] = b[i]["sft_qwen"]
    
    # with open(path3, 'w') as f:
    #     json.dump(c, f, indent=4)
    
