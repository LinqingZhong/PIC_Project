import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.simplefilter("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import time


model_list = ["internlm-3-8B", "internlm-2.5-7B"]
model_name = None
while model_name not in model_list:
    model_name = input(f"Please choose the desired model in {', '.join(model_list)}]: ")
    model_name = model_name.strip()
    if model_name not in model_list:
        print("Invalid model name. Re-enter.")

if model_list.index(model_name) == 0:
    model_dir = "../ckpts/internlm-3-8B"
else:
    model_dir = "../ckpts/internlm-2_5-7B"
print(f"Loading the model {model_name}......")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model = model.eval()

system_prompt = ""
while True:
    prompt = input("\nPlease answer your question, enter 'quit' to leave:")
    start = time.time()
    prompt = prompt.strip()
    if prompt == "quit":
        print("Assistant: Looking forward to see you again.")
        print("Ending the process......")
        break
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
    ]
    prompt = tokenizer.batch_decode(tokenized_chat)[0]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end = time.time()
    print("Assistant: ", response)
    print(f"Inference time: {end - start} s.")