import torch
import transformers
import warnings
warnings.simplefilter("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import time


model_name = "Llama-3.1-8B"
print(f"The model that will be loaded is {model_name}.")
print(f"Loading the model {model_name}......")

model = transformers.pipeline(
    "text-generation",
    model="/mnt/data4/zlq/pkgs/llama-3.1-8B",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    # flash_attn = False
)


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
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]

    outputs = model(messages, max_new_tokens=4096,)
    response = outputs[0]["generated_text"][-1]['content']
    end = time.time()
    print("Assistant: ", response)
    print(f"Inference time: {end - start} s.")