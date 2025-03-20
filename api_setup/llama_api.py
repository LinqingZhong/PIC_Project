from server_wrapper import ServerMixin, host_model, send_request, str_to_image
import transformers
import torch

class Llama3_1(object):
    
    def __init__(self, model_path="../ckpts/llama-3.1-8B"):
        self.model = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )


    def predict(self, prompt):

        messages = [
            {"role": "system", "content": "You are an intelligent assistant whose name is Llama-3.1. You need to answer question carefully."},
            {"role": "user", "content": prompt},
        ]

        outputs = self.model(
            messages,
            max_new_tokens=4096,
        )
        return outputs[0]["generated_text"][-1]['content']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12188)
    args = parser.parse_args()

    print("Loading model...")

    class LlamaServer(ServerMixin, Llama3_1):
        def process_payload(self, payload: dict):
            prompt = payload["caption"]
            answer = self.predict(prompt)
            return answer

    llama_api = LlamaServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(llama_api, name="llama", port=args.port, IP = "115.25.142.41") # change the ip according to your computer