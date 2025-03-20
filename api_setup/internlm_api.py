from server_wrapper import ServerMixin, host_model, send_request, str_to_image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Internlm(object):
    
    # def __init__(self, model_path="../ckpts/internlm-2_5-7B"):
    def __init__(self, model_path="../ckpts/internlm-3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.model = self.model.eval()


    def predict(self, prompt):

        system_prompt = ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
        ]
        prompt_encode = self.tokenizer.batch_decode(tokenized_chat)[0]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12188)
    args = parser.parse_args()

    print("Loading model...")

    class InternlmServer(ServerMixin, Internlm):
        def process_payload(self, payload: dict):
            prompt = payload["caption"]
            return self.predict(prompt)

    Internlm_api = InternlmServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(Internlm_api, name="internlm", port=args.port, IP = "115.25.142.41") # change the ip according to your computer