import json
import re
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from tqdm import tqdm
llm=Ollama(model="qwen2.5:14b",temperature=0.7,request_timeout=60)

def read_from_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        samples = json.load(f)
    return samples
def extract_think_and_answer(text):
    # 定义正则表达式模式来匹配 <think></think> 标签及其内容
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    # 查找 <think></think> 标签内的内容
    think_match = think_pattern.search(text)
    think_content = think_match.group(1).strip() if think_match else ""
    # 移除 <think></think> 标签及其内容
    answer_content = think_pattern.sub('', text).strip()
    return think_content,answer_content
#将thinking中参考去掉，重新组织
SYSTEM_INSTRUCT_FOR_THINK="""
你的任务是修改文本，但不改变文本的原意。保留对问题的描述和语气，去掉文本中关于参考的描述,将其作为"我"的所见所闻所知，保持推理过程不变
"""
THINK_PROMPT="""
修正以下文本中的思考过程。直接给出修正后的结果。
给定文本：
{content}
修正后的文本:
"""
SYSTEM_INSTRUCT_FOR_ANSWER="""
你的任务是根据用户的提问，推理过程和原有的回答，重新组织回答。保持回答的准确性、详细性。
"""
SUMMARY_ANSWER_PROMPT="""
直接给出重新组织的回答，不要解释。
用户提问:{question}
根据推理过程：{think},
原有的回答：{origin_answer}
重新组织回答：
"""
THINK_FORMAT="<think>\n{think_content}\n</think>\n"
ANSWER_FORMAT="<answer>\n{answer_content}\n</answer>\n"
def modifyAgent(system_instruct,user_content):
    resp=llm.chat(messages=[
        ChatMessage(content=system_instruct,role="system"),
        ChatMessage(content=user_content,role="user")
    ])
    return resp.message.content

filepath="distill-cog_10.json"
filename=os.path.basename(filepath)
savepath=f"modify_{filename}"
samples=read_from_json(filepath)
mSamples=[]
if os.path.exists(savepath):
    mSamples=read_from_json(savepath)
alreadyIds=[sample["id"] for sample in mSamples]
samples=[sample for sample in samples if sample["id"] not in alreadyIds]
for sample in tqdm(samples):
    num=len(sample["messages"])
    id=sample["id"]
    if "excavation_image_text_summarize" in id:
        continue
    for i in range(num//2):
        user_content=sample["messages"][i*2]["content"]
        assistant_content=sample["messages"][i*2+1]["content"]
        think,answer=extract_think_and_answer(assistant_content)
        print("**"*20,"origin","**"*20)
        print(think)
        print("=="*20,"modify","=="*20)
        mthink=modifyAgent(SYSTEM_INSTRUCT_FOR_THINK,THINK_PROMPT.format(content=think))
        manswer=modifyAgent(SYSTEM_INSTRUCT_FOR_ANSWER,SUMMARY_ANSWER_PROMPT.format(question=user_content,think=mthink,origin_answer=answer))
        
        new_content=THINK_FORMAT.format(think_content=mthink)+manswer
        print(THINK_FORMAT.format(think_content=mthink)+"\n"+ANSWER_FORMAT.format(answer_content=manswer))
        sample["messages"][i*2+1]["content"]=new_content
    mSamples.append(sample)
    with open(savepath,'w',encoding="utf-8") as f:
        json.dump(mSamples,f,ensure_ascii=False,indent=4)
