import fitz
import json
import time
import re
from openai import OpenAI

client = OpenAI(
    api_key="sk-yw7wudjjTRjskZTlF0Df76A1B3824eB1Ac1307A354DeBd03",
    base_url="https://api3.apifans.com/v1"
)

def extract_text_from_pdf(pdf_path, output_path, start_page, end_page):
    doc = fitz.open(pdf_path)
    text_all = ""
    for page_number in range(len(doc)):
        if page_number < start_page or page_number > end_page:
            continue
        text = doc.load_page(page_number).get_text("text")
        text = text.strip().replace("-\n", "")
        text_all += text

    text_per_line = text_all.split("\n")
    text_per_par = []
    
    par = ""
    for i, text in enumerate(text_per_line):
        if not par.endswith(".") and not par.endswith("?"):
            par = par + " " +text
        else:
            text_per_par.append(par.strip())
            par = ""
        
    with open(output_path, "w") as f:
        json.dump(text_per_par, f, indent=4)

def generate_qa_per_par(par):
    global client
    qa_pairs = []
    prompt = "Please generate as many meaningful question-answer pairs as possible based on the following text."
    prompt += "Here are some requirements:"
    prompt += "\n1. Cover important facts, concepts, and details from the text."
    prompt += "\n2. Questions should be specific and informative—not too broad or vague."
    prompt += "\n3. Format your response as a list. Each element should be a dictionary with two fields: 'question' and 'answer'."
    prompt += f"Here is the text:\n{par}"

    completion = client.chat.completions.create(
        messages=[
        {"role": "user", "content": prompt}
    ],
        model="gpt-4o",
    )

    content = completion.choices[0].message.content
    pattern = re.compile(r'"?question"?\s*:\s*["“](.*?)["”]\s*,?\s*"answer"?\s*:\s*["“](.*?)["”]', re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(content)

    for q, a in matches:
        qa_pairs.append({
            "question": q.strip(),
            "answer": a.strip()
        })
    return qa_pairs

def get_qa_pairs(text_path, output_path, limit=300):
    global client

    with open(text_path, 'r') as f:
        data = json.load(f)
    qa_pairs = []

    for i, par in enumerate(data):
        qa_result = generate_qa_per_par(par.strip())
        qa_pairs.extend(qa_result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4)


def get_pure_qa_pairs(data_path, output_path, save_step = 100):
    global client
    with open(data_path, 'r') as f:
        data = json.load(f)

    data_pure = []
    for i in range(len(data)):
        item = data[i]
        question = item["question"]
        answer = item["answer"]
        prompt = "I need to collect some question-answer pairs on the Oil and Gas Industry. The question-answer pair should be related to the oil and gas field, such as petroleum, chemicals and drilling, and should be professional enough."
        prompt += "Here is a example: What is oilsand? Oilsand is one of the forms of unconventional oil that is commercially recovered to produce bitumen as an oil product."
        prompt += "Now I give you a candidate question-answer pair. You need to determine whether this question-answer pair meets the above requirements, that is, whether it is relevant to the energy field and professional enough."
        prompt += f"The question is: {question}" + f"\nThe answer to this question is: {answer}"
        prompt += "Answer 'True' if it meets the requirement. Otherwise, answer 'False'."

        try:
            completion = client.chat.completions.create(
                messages=[
                {"role": "user", "content": prompt}
            ],
                model="gpt-4o",
            )

            content = completion.choices[0].message.content
            if 'true' in content.lower():
                data_pure.append({"question": question, "answer": answer})
        except:
            time.sleep(3)

        if i % save_step == 0 and i != 0:
            print(f"{str(len(data_pure))} / {str(i)}")
            with open(output_path.replace(".json", "_"+str(i // save_step)+".json"), 'w') as f:
                json.dump(data_pure, f, indent=4)
