from server_wrapper import ServerMixin, host_model, send_request, str_to_image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen2_5(object):
    
    def __init__(self, model_path="../ckpts/Qwen2.5-7B"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, prompt):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12188)
    args = parser.parse_args()

    print("Loading model...")

    class Qwen2_5_Server(ServerMixin, Qwen2_5):
        def process_payload(self, payload: dict):
            prompt = payload["caption"]
            return self.predict(prompt)

    Qwen2_5_api = Qwen2_5_Server()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(Qwen2_5_api, name="qwen", port=args.port, IP = "115.25.142.41") # change the ip according to your computer