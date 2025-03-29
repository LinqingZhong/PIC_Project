import os
import json
import jsonlines
import torch
import copy
import random
from torch.utils.data import Dataset
from typing import Dict, List
from openai import OpenAI
from functools import partial
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq
from transformers import Trainer, BitsAndBytesConfig,TrainingArguments,set_seed,Seq2SeqTrainer 
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path to the model")
argparser.add_argument("--train_file_or_dir", type=str, nargs='+', required=True, help="Path to training file")
argparser.add_argument("--val_file_or_dir", type=str, default=None, help="Path to validation file")
argparser.add_argument("--schema_file", type=str,nargs='+',required=True, help="Path to schema file to build the system_message")
argparser.add_argument("--output_dir", type=str, required=True)
argparser.add_argument("--learning_rate", type=float, required=True)
argparser.add_argument("--model_max_length", type=int, required=True)
argparser.add_argument("--max_steps", type=int, required=True)
argparser.add_argument("--train_batch_size", type=int, required=True)
argparser.add_argument("--eval_batch_size", type=int)
argparser.add_argument("--gradient_accumulation_steps", type=int, required=True)
argparser.add_argument("--log_step", type=int, required=True)
argparser.add_argument("--evaluation_strategy", type=str)
argparser.add_argument("--save_strategy", type=str, required=True)
argparser.add_argument("--save_steps", type=int, required=True)
argparser.add_argument("--compute_type", type=str, required=True)
argparser.add_argument("--deepspeed", type=str, default=None)
argparser.add_argument("--lora_r", type=int, required=True)
argparser.add_argument("--lora_alpha", type=float, required=True)
argparser.add_argument("--lora_dropout", type=float, required=True)
argparser.add_argument("--lora_target_modules", nargs='+', required=True)
argparser.add_argument("--openai_api_base", type=str)
argparser.add_argument("--openai_api_key", type=str)
args = argparser.parse_args()

def build_train_args(args):
    train_args = TrainingArguments(
        do_train=True,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        # lr_scheduler_type="constant",
        logging_dir=os.path.join(args.output_dir,"log"),
        logging_steps=args.log_step,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if args.deepspeed is not None:
        train_args.deepspeed=args.deepspeed
    if args.val_file_or_dir is not None:
        train_args.per_device_eval_batch_size=args.eval_batch_size,
        train_args.evaluation_strategy=args.evaluation_strategy
    return train_args

def build_messages(schema,question,lang):

    if lang=="ENG":
        prompt_sys=f"""You are an experienced Cypher developer and helpful medical expert.

        Your task is to answer the medical questions by generating the corresponding Cypher statements.

        You will be given the Neo4j graph database with the following schema to look for the necessary information.
        {schema}

        Please only include the cypher in your response. Do not add any comments.
        """
        prompt_user=f"""Here are some examples of the Question-Cypher pairs as reference.

        <|Example 1|>
        Question: "What are the diseases that commonly accompany 'Depression'?",
        Cypher: "MATCH (d1:Disease)-[:acompany_with]->(d2:Disease) WHERE d1.name='Depression' RETURN d2.name"

        You should answer the question following the way reference examples do.
        You must only include the cypher in your response. Do not add any comments.

        Now it is your turn to answer the question by generating the cypher!
        Question: {question}
        Cypher:
        """

    elif lang=="CN":
        prompt_sys=f"""You are an experienced Cypher developer, a Chinese and English bilingual medical expert.

        Your task is to answer the medical questions by generating the corresponding Cypher statements.

        You will be given the Neo4j graph database with the following schema to look for the necessary information.
        {schema}

        Please only include the cypher in your response. Do not add any comments.
        """

        prompt_user=f"""Here are some examples of the Question-Cypher pairs as reference.

        <|Example 1|>
        Question: "What are the diseases that commonly accompany '抑郁症'?",
        Cypher: "MATCH (d1:Disease)-[:acompany_with]->(d2:Disease) WHERE d1.name='抑郁症' RETURN d2.name"

        You should answer the question following the way reference examples do.
        You must only include the cypher in your response. Do not add any comments before or after the cypher.

        Now it is your turn to answer the question by generating the cypher!
        Question: {question}
        Cypher:
        """

    # if lang=="ENG":
    #     prompt_sys=f"""You are an experienced Cypher developer and helpful medical expert.
    #     Your task is to answer the medical questions by generating the corresponding Cypher statements.
    #     """
    #     prompt_user=f"""{question}"""

    # elif lang=="CN":
    #     prompt_sys=f"""You are an experienced Cypher developer, a Chinese and English bilingual medical expert.
    #     Your task is to answer the medical questions by generating the corresponding Cypher statements.
    #     """
    #     prompt_user=f"""{question}"""
    
    # prompt_sys=f"""Answer the question according to the schema below:
    # {schema}"""
    # prompt_user=f"""{question}"""

    return prompt_sys,prompt_user

def internlm_preprocess(data,tokenizer,max_len,schema,lang):
    datas=[]
    template=(
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    sources,targets,full_conversations=[],[],[]
    for item in data:
        instr,input=build_messages(schema,item["question"],lang)
        output=item["cypher"]
        sources.append(template.format_map({'instruction': instr+"\n", 'input': input+"\n"})[:max_len])
        targets.append(f"{output[:max_len-1]}{tokenizer.eos_token}")
    
    for source,target in zip(sources,targets):
        full_conversations.append(source+target)

    def tokenize_fn(datas,tokenizer):
        tokenized_list=[]
        for data in datas:
            tokenized_list.append(tokenizer(data,return_tensors="pt",padding="longest",max_length=tokenizer.model_max_length,truncation=True,))
        input_ids=labels=[tokenized_data.input_ids[0] for tokenized_data in tokenized_list]
        ne_pad_token_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        input_ids_lens = labels_lens = [tokenized_data.input_ids.ne(ne_pad_token_id).sum().item() for tokenized_data in tokenized_list]
        return dict(input_ids=input_ids,labels=labels,input_ids_lens=input_ids_lens,labels_lens=labels_lens)

    sources_tokenized=tokenize_fn(sources,tokenizer)
    full_conversations_tokenized=tokenize_fn(full_conversations,tokenizer)

    input_ids = full_conversations_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = -100
    assert len(input_ids)==len(labels)
    for i in range(len(input_ids)):
        datas.append({"input_ids":input_ids[i],"labels":labels[i]})
    return datas

def llama_qwen_preprocess(data,tokenizer,max_len,schema,lang):
    MAX_LENGTH = max_len
    datas=[]
    for item in data:
        system_message,user_message=build_messages(schema,item["question"],lang)
        cypher=item["cypher"]
        # example={"instruction":system_message,"input":user_message,"output":item["cypher"]}    
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        response = tokenizer(f"{cypher}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] 
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        datas.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })
    return datas

def glm_preprocess(messages,tokenizer,max_len,schema,lang):
        items=[]
        for i,item in enumerate(messages):
            system_message,user_message=build_messages(schema,item["question"],lang)
            input=system_message+"\n"+user_message
            output=item["cypher"]
            if len(output)==0:
                continue
            a_ids = tokenizer.encode(text=input, add_special_tokens=True, truncation=True,
                                        max_length=max_len)
            b_ids = tokenizer.encode(text=output, add_special_tokens=False, truncation=True,
                                        max_length=512)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            labels[labels == tokenizer.pad_token_id] = -100
            items.append({"input_ids": input_ids,"labels": labels})
        return items

class T2CDataset(Dataset):
    def __init__(self, model_name, data_list, schema_file_list, lang_list, tokenizer, max_len, ignored_key=-100):
        super(T2CDataset, self).__init__()
        self.model_name=model_name
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ignored_key = ignored_key
        self.data=[]
        assert len(data_list)==len(schema_file_list)
        assert len(data_list)==len(lang_list)
        for i in range(len(data_list)):
            data=data_list[i]
            schema_file=schema_file_list[i]
            lang=lang_list[i]
            schema = ""
            with open(schema_file, 'r') as f:
                schema = f.read()

            if "glm" in self.model_name.lower():
                self.data+=glm_preprocess(data,tokenizer,self.max_len,schema,lang)
            elif "qwen" in self.model_name.lower() or 'llama' in self.model_name.lower():
                self.data+=llama_qwen_preprocess(data,tokenizer,max_len,schema,lang)
            elif "internlm" in self.model_name.lower():
                self.data+=internlm_preprocess(data,tokenizer,max_len,schema,lang)
        random.shuffle(self.data)
        # print(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict:
        return self.data[i]

train_data_list=[]
for train_file_or_dir in args.train_file_or_dir:
    train_data=[]
    if os.path.isfile(train_file_or_dir):
        with open(train_file_or_dir,'r') as f:
            a=json.load(f)
            for item in a:
                train_data.append({"question":item["question"],"cypher":item["cypher"]})
    elif os.path.isdir(train_file_or_dir):
        for file_name in os.listdir(train_file_or_dir):
            if file_name.endswith(".json"):
                f=open(os.path.join(train_file_or_dir,file_name),'r')
                try:
                    item=json.load(f)
                    train_data.append({"question":item["question"],"cypher":item["cypher"]})
                    f.close()
                except Exception as e:
                    f.close()
    train_data_list.append(copy.deepcopy(train_data))

if args.val_file_or_dir is not None:
    val_data_list=[]
    for val_file_or_dir in args.val_file_or_dir:
        val_data=[]
        if os.path.isfile(val_file_or_dir):
            with open(val_file_or_dir,'r') as f:
                a=json.load(f)
                for item in a:
                    val_data.append({"question":item["question"],"cypher":item["cypher"]})
        elif os.path.isdir(val_file_or_dir):
            for file_name in os.listdir(val_file_or_dir):
                if file_name.endswith(".json"):
                    f=open(os.path.join(val_file_or_dir,file_name),'r')
                    try:
                        item=json.load(f)
                        val_data.append({"question":item["question"],"cypher":item["cypher"]})
                        f.close()
                    except Exception as e:
                        f.close()
        val_data_list.append(copy.deepcopy(val_data))

lang_list=[]
for i in range(len(args.schema_file)):
    if 'hetionet' in args.schema_file[i].lower():
        lang_list.append("ENG")
    elif 'lhy' in args.schema_file[i].lower():
        lang_list.append('CN')

if "openai" in args.model_name_or_path.lower():

    model_name=args.model_name_or_path.split("/")[-1]
    api_url=args.openai_api_base
    api_key=args.openai_api_key
    # client=OpenAI(api_key=api_key, base_url=api_url)
    client=OpenAI(api_key=api_key)
    assert len(train_data_list)==len(args.schema_file)
    assert len(train_data_list)==len(lang_list)

    GPT_train_file=os.path.join(args.output_dir,"SFT_GPT_train.jsonl")
    if os.path.exists(GPT_train_file):
        os.remove(GPT_train_file)
    with jsonlines.open(GPT_train_file,'a') as f:
        for i in range(len(train_data_list)):
            train_data=train_data_list[i]
            schema_file=args.schema_file[i]
            lang=lang_list[i]
            schema = ""
            with open(schema_file, 'r') as g:
                schema = g.read()
            
            for item in train_data:
                system_message,user_message=build_messages(schema,item['question'],lang)
                message={"messages":[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}, {"role": "assistant", "content": item['cypher']}]}
                f.write(message)
    print("Created SFT file")
    for retry in range(20):
        try:
            GPT_train_file_id=client.files.create(file=open(GPT_train_file, "rb"),purpose="fine-tune").id
            print(f"SFT file id is {GPT_train_file_id}")
            break
        except Exception as e:
            print(f"Retrying {retry+1}/20")
    if args.val_file_or_dir is not None:
        GPT_val_file=os.path.join(args.output_dir,"SFT_GPT_val.json")
        with open(GPT_val_file,'a') as g:
            for i in range(len(val_data_list)):
                val_data=val_data_list[i]
                schema_file=args.schema_file[i]
                lang=lang_list[i]
                schema = ""
                with open(schema_file, 'r') as h:
                    schema = h.read()
                for item in val_data:
                    system_message,user_message=build_messages(schema,item['question'],lang)
                    message={"messages":[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}, {"role": "assistant", "content": item['cypher']}]}
                    g.write(message)
        GPT_val_file_id=client.files.create(file=open(GPT_val_file, "rb"),purpose="fine-tune").id
        client.fine_tuning.jobs.create(model=model_name,training_file=GPT_train_file_id,validation_file=GPT_val_file_id)
    else:
        for retry in range(20):
            try:
                client.fine_tuning.jobs.create(model=model_name,training_file=GPT_train_file_id)
                print("Created openai finetune job")
                break
            except Exception as e:
                print(f"Retrying {retry+1}/20")
else:

    set_seed(24)
    if args.compute_type=="fp16":
        compute_dtype = (torch.float16)
    elif args.compute_type=="bf16":
        compute_dtype = (torch.bfloat16)
    else:
        compute_dtype = torch.float32

    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=compute_dtype)

    config = AutoConfig.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,model_max_length=args.model_max_length,trust_remote_code=True)
    if "llama" in args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                config=config,
                                                torch_dtype=torch.bfloat16,
                                                # device_map = 'cuda',
                                                trust_remote_code=True,)
                                                #  quantization_config=quantization_config)
    # model = prepare_model_for_kbit_training(model)

    train_set=T2CDataset(args.model_name_or_path,train_data_list,args.schema_file,lang_list,tokenizer,args.model_max_length,args.model_max_length)
    if args.val_file_or_dir is not None:
        val_set=T2CDataset(args.model_name_or_path,val_data_list,args.schema_file,lang_list,tokenizer,args.model_max_length,args.model_max_length)

    if 'glm' not in args.model_name_or_path.lower():
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True
        )
    loraconfig = LoraConfig(
        task_type="CAUSAL_LM", 
        target_modules=args.lora_target_modules,
        inference_mode=False,r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model=get_peft_model(model, loraconfig)
    model.print_trainable_parameters()

    if args.val_file_or_dir is not None:
        trainer = Trainer (
            model=model,
            tokenizer=tokenizer if 'glm' not in args.model_name_or_path.lower() else None,
            args=build_train_args(args),
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer if 'glm' not in args.model_name_or_path.lower() else None,
            args=build_train_args(args),
            train_dataset=train_set,
            data_collator=data_collator
        )
    trainer.train()
    trainer.save_model()