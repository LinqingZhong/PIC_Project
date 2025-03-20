import os
os.makedirs("./ckpts/llama-3.1-8B", exist_ok=True)
os.makedirs("./ckpts/internlm-2.5-7B", exist_ok=True)
os.makedirs("./ckpts/Qwen2.5-7B", exist_ok=True)
os.makedirs("./ckpts/internlm-3-8B", exist_ok=True)


from modelscope import snapshot_download 
model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', local_dir = "/llama-3.1-8B/")
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2_5-7b-chat', local_dir = "./ckpts/internlm-2.5-7B/")
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir = "./ckpts/Qwen2.5-7B/")
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct', local_dir = "./ckpts/internlm-3-8B/")
