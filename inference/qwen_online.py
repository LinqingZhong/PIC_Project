import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.simplefilter("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import time

model_name = "Qwen-2.5-7B"
print(f"The model that will be loaded is {model_name}.")
print(f"Loading the model {model_name}......")

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/data4/zlq/pkgs/Qwen2.5-7B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/data4/zlq/pkgs/Qwen2.5-7B")

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
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end = time.time()
    print("Assistant: ", response)
    print(f"Inference time: {end - start} s.")