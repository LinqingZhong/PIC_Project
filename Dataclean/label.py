import json
from openai import AzureOpenAI
import os
import time

def initialize_client():
    endpoint = os.getenv("ENDPOINT_URL", "your_url")
    deployment = os.getenv("DEPLOYMENT_NAME", "name")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "your_api_key")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )
    return client, deployment

def load_input_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return [{"role": "system", "content": f.read()}]

def generate_qa_pairs(client, deployment, text_content, system_prompt):
    messages = system_prompt + [{"role": "user", "content": text_content}]
    
    for _ in range(3):  # Retry机制
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                top_p=0.9
            )
            # 确保返回的内容是 JSON 格式
            res = response.choices[0].message.content
            try:  
                # 尝试解析为 JSON  
                qa_pairs = json.loads(res)  
                return qa_pairs  
            except json.JSONDecodeError:  
                print("返回内容不是有效的 JSON 格式，尝试修复...")  
                print("返回内容：", res)  # 输出返回内容以便调试  
                # 如果返回内容不是 JSON，尝试修复格式  
                if res.startswith("```json") and res.endswith("```"):  
                    res = res[7:-3].strip()  # 去掉 ```json 和 ```  
                try:  
                    qa_pairs = json.loads(res)  
                    return qa_pairs  
                except json.JSONDecodeError as e:  
                    print(f"JSON 解析失败: {e}")  
                    print("修复后的内容：", res)  # 输出修复后的内容以便调试  
                    return []  
        except Exception as e:  
            print(f"API错误: {str(e)}")  
            time.sleep(2)  
    return [] 

def process_all_texts(client, deployment, input_data, system_prompt):
    all_qa = []
    for item in input_data["texts"]:
        text = item["text"]
        qa_pairs = generate_qa_pairs(client, deployment, text, system_prompt)
        if qa_pairs:
            all_qa.extend(qa_pairs)
        else:
            print(f"文本处理失败: {text[:50]}...")
    return all_qa

def save_output(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 初始化配置
    client, deployment = initialize_client()
    system_prompt = load_prompt("rule.prompt")
    
    # 加载输入数据
    input_data = load_input_data("data/Future_energy.json")
    # input_data = load_input_data("data/input.json")
    
    # 处理所有文本
    qa_data = process_all_texts(client, deployment, input_data, system_prompt)
    
    # 保存结果
    save_output(qa_data, "../SFT_Dataset.json")
    print("处理完成")