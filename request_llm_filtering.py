from openai import OpenAI
import json
import time

client = OpenAI(
    api_key="sk-yw7wudjjTRjskZTlF0Df76A1B3824eB1Ac1307A354DeBd03",
    base_url="https://api3.apifans.com/v1"
)

path = "/mnt/data4/zlq/PIC_Data/data4.json" # vanilla json path
with open(path, 'r') as f:
    data1 = json.load(f)

data_pure = []
save_step = 100
for i in range(len(data1)):
    item = data1[i]
    question = item["instruction"]
    answer = item["output"]
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
        continue

    if i%save_step == 0 and i!=0:
        print(f"{str(len(data_pure))} / {str(i)}")
        with open(f"/mnt/data4/zlq/PIC_Data/SFT_Dataset.json", 'w') as f: # output json path
            json.dump(data_pure, f, indent=4)