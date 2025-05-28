import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_SDP_FORCE_DISABLE"] = "1"
import json
import argparse
import datetime
import evaluate
import torch
from torch.utils.data import Dataset

from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq


def load_json_file(json_path):
    with open(json_path, 'r') as f:
        result = json.load(f)
    return result

class LLamaDataset(Dataset):
    
    def __init__(self, raw_data, tokenizer, sys_prompt = None, instruction_prompt =None, max_length = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sys_prompt = sys_prompt
        self.instruction_prompt = instruction_prompt
        self._preprocess(raw_data)
    
    def __len__(self):
        return len(self.data)
    
    def _preprocess(self, raw_data):
        self.data = []

        for item in raw_data:
            question = item["question"]
            answer = item["answer"]
            sentence = self.tokenizer.bos_token
            if self.sys_prompt is not None:
                sentence = sentence + "<|start_header_id|>system<|end_header_id|>\n\n" + self.sys_prompt + self.tokenizer.eos_token
            if self.instruction_prompt is not None:
                sentence = sentence + "<|start_header_id|>user<|end_header_id|>\n\n" + self.instruction_prompt + question + self.tokenizer.eos_token
            else:
                sentence = sentence + "<|start_header_id|>user<|end_header_id|>\n\n" + question + self.tokenizer.eos_token
            sentence = sentence + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            answer = item["answer"] + self.tokenizer.eos_token
            
            inputs = self.tokenizer(sentence, add_special_tokens = False)
            outputs = self.tokenizer(answer, add_special_tokens = False)
            input_ids = inputs["input_ids"] + outputs["input_ids"]
            labels = [-100] * len(inputs["input_ids"]) + outputs["input_ids"]
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            single_item = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
            self.data.append(single_item)
            
    
    def __getitem__(self, index):
        return self.data[index]

class QwenDataset(LLamaDataset):
    def __init__(self, raw_data, tokenizer, sys_prompt = None, instruction_prompt =None, max_length = 4096):
        self.PROMPT_DICT = {
            "prompt_no_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n""",
            "prompt_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n""",
        }
        LLamaDataset.__init__(self, raw_data, tokenizer, sys_prompt, instruction_prompt, max_length)

    def _preprocess(self, raw_data):
        self.data = []

        for item in raw_data:
            question = item["question"]
            answer = item["answer"]
            if self.instruction_prompt is not None:
                sentence = self.PROMPT_DICT["prompt_input"].format_map({"instruction": self.sys_prompt, "input": self.instruction_prompt + question})
            else:
                sentence = self.PROMPT_DICT["prompt_input"].format_map({"instruction": self.sys_prompt, "input": question})
            answer = answer.strip() + self.tokenizer.eos_token

            full_inputs = self.tokenizer(sentence + answer, return_tensors="pt", add_special_tokens = False)
            inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens = False)
            
            input_ids = full_inputs["input_ids"][0]
            labels = input_ids.clone()

            idx_to_neglect = inputs["input_ids"][0].ne(self.tokenizer.pad_token_id).sum().item()
            labels[:idx_to_neglect] = -100

            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            single_item = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
            self.data.append(single_item)

class InternLMDataset(LLamaDataset):
    def __init__(self, raw_data, tokenizer, sys_prompt = None, instruction_prompt =None, max_length = 4096):
        LLamaDataset.__init__(self, raw_data, tokenizer, sys_prompt, instruction_prompt, max_length)
    
    def _preprocess(self, raw_data):
        self.data = []
        for item in raw_data:
            question = item["question"]
            answer = item["answer"]
            sentence = self.tokenizer.bos_token
            if self.sys_prompt is not None:
                sentence = sentence + f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            if self.instruction_prompt is not None:
                sentence = sentence + f"<|im_start|>user\n{self.instruction_prompt + question}<|im_end|>\n"  
            else:
                sentence = sentence + f"<|im_start|>user\n{question}<|im_end|>\n"  
            sentence = sentence + f"<|im_start|>assistant\n"
            answer = answer + "<|im_end|>\n" + self.tokenizer.eos_token

            inputs = self.tokenizer(sentence, add_special_tokens = False)
            outputs = self.tokenizer(answer, add_special_tokens = False)

            input_ids = inputs["input_ids"] + outputs["input_ids"]
            labels = [-100] * len(inputs["input_ids"]) + outputs["input_ids"]
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            single_item = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
            self.data.append(single_item)


class LLMTrainer(object):
    
    def __init__(self, json_path):
        self._load_config(json_path)
        self._load_model()
        self._load_dataset()
        self._load_training_args()
        self._get_lora_model()

    def _load_config(self, json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
        self.root_dir = config["root_dir"]
        self.model_type = config["model_type"]
        self.model_name_or_path = config["model_name_or_path"]
        self.freeze_embedding = config["freeze_embedding"]
        self.metrics = config["metrics"]
        
        self.max_length = config["data_config"]["max_length"]
        self.sys_prompt = config["data_config"]["system_prompt"]
        self.instruction_prompt = config["data_config"]["instruction_prompt"]
 
        self.training_data = load_json_file(config["data_config"]["training_json_path"])
        self.val_data = load_json_file(config["data_config"]["eval_json_path"])
        
        self.lora_config = config["lora_config"]
        self.training_config = config["training_config"]

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False, trust_remote_code=True, local_files_only=True )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
        if not self.freeze_embedding:
            self.model.enable_input_require_grads()

    def _load_dataset(self):
        if self.model_type == "llama":
            self.training_dataset = LLamaDataset(self.training_data, self.tokenizer, self.sys_prompt, self.instruction_prompt, self.max_length)
            self.val_dataset = LLamaDataset(self.val_data, self.tokenizer, self.sys_prompt, self.instruction_prompt, self.max_length)
        elif self.model_type == "qwen":
            self.training_dataset = QwenDataset(self.training_data, self.tokenizer, self.sys_prompt, self.instruction_prompt, self.max_length)
            self.val_dataset = QwenDataset(self.val_data, self.tokenizer, self.sys_prompt, self.instruction_prompt, self.max_length)
        elif self.model_type == "internlm":
            self.training_dataset = InternLMDataset(self.training_data, self.tokenizer, self.sys_prompt, self.instruction_prompt, self.max_length)
            self.val_dataset = InternLMDataset(self.val_data, self.tokenizer, self.sys_prompt, self.instruction_prompt, self.max_length)
        else:
            raise NotImplementedError
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True, return_tensors="pt")

    def _load_training_args(self):
        exp_dir = os.path.join(self.root_dir, self.training_config["run_name"])
        if os.path.exists(exp_dir):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(self.root_dir, f"{self.training_config['run_name']}_{timestamp}")
        else:
            os.makedirs(exp_dir)
        self.training_args = TrainingArguments(
            output_dir = exp_dir,
            overwrite_output_dir = self.training_config["overwrite_output_dir"],
            num_train_epochs = self.training_config["num_train_epochs"],
            per_device_train_batch_size = self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size = self.training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps = self.training_config["gradient_accumulation_steps"],
            evaluation_strategy = self.training_config["evaluation_strategy"],
            save_strategy = self.training_config["save_strategy"],
            logging_steps = self.training_config["logging_steps"],
            eval_steps = self.training_config["eval_steps"],
            save_steps = self.training_config["save_steps"],
            learning_rate = self.training_config["learning_rate"],
            weight_decay = self.training_config["weight_decay"],
            bf16 = self.training_config["bf16"],
            logging_dir = os.path.join(exp_dir, "logs"),
            lr_scheduler_type = self.training_config["lr_scheduler_type"],
            warmup_ratio = self.training_config["warmup_ratio"],
            report_to = "tensorboard"
        )
        print(self.training_args)
        # self.training_args.predict_with_generate = True
        # self.training_args.generation_max_length = self.max_length

    def _get_lora_model(self):
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            inference_mode = False,
            r = self.lora_config["r"],
            lora_alpha = self.lora_config["lora_alpha"],
            lora_dropout = self.lora_config["dropout"],
            target_modules = self.lora_config["target_modules"]
        )
        self.model = get_peft_model(self.model, peft_config)

    def _compute_metrics(self, eval_preds):
        predictions, labels = eval_preds

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        results = {}

        if "bleu" in self.metrics:
            bleu = evaluate.load("bleu")
            bleu_preds = [pred.split() for pred in decoded_preds]
            bleu_labels = [[label.split()] for label in decoded_labels]
            results["bleu"] = bleu.compute(predictions=bleu_preds, references=bleu_labels)["bleu"]

        if "meteor" in self.metrics:
            meteor = evaluate.load("meteor")
            results["meteor"] = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]

        if "rouge" in self.metrics:
            rouge = evaluate.load("rouge")
            rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            results["rouge1"] = rouge_score["rouge1"]
            results["rouge2"] = rouge_score["rouge2"]
            results["rougeL"] = rouge_score["rougeL"]
            results["rougeLsum"] = rouge_score["rougeLsum"]

        return results
    
    def run(self):
        trainer = Trainer(
        # trainer = Seq2SeqTrainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.training_dataset,
            eval_dataset = self.val_dataset,
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            # compute_metrics = self._compute_metrics
        )
        self.model.print_trainable_parameters()
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    runner = LLMTrainer(args.config)
    runner.run()
    print("Training Finished!")
