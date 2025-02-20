import json
import openai
import os
import time
import asyncio
import backoff
import re


def read_from_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        samples = json.load(f)
    return samples

ANSWER_FORMAT="<answer>{answer}</answer>"
def extract_think_and_answer(text):
    # 定义正则表达式模式来匹配 <think></think> 标签及其内容
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    # 查找 <think></think> 标签内的内容
    think_match = think_pattern.search(text)
    think_content = think_match.group(1).strip() if think_match else ""
    # 移除 <think></think> 标签及其内容
    answer_content = think_pattern.sub('', text).strip()
    return {
        "think": think_content,
        "answer": answer_content
    }

class Distiller:
    def __init__(self, api_key, base_url, model="deepseek-chat"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def create_messages(self, user_prompt, label_output,history=[]):
        content = (f"结合参考以及你已有的知识回答问题：{user_prompt}"
                   f"参考:{label_output}")
        messages=[{"role": "system", "content": "You are a helpful assistant."}]
        for hist in history:
            messages.append(hist)
        messages.append({"role": "user", "content": content})
        return messages

    async def infer_single(self, user_prompt, label_output,history=[]):
        messages = self.create_messages(user_prompt, label_output,history)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                stream=False,
                max_tokens=4096
            )
            resp_content=response.choices[0].message.content
            #answer=get_answer(resp_content)
            result=[{"role":"user","content":user_prompt},{"role":"assistant","content":resp_content}]
            return result
        except openai.APIError as e:
            print(f"OpenAI API Error in evaluation: {e}")
        except Exception as e:
            print(f"Other error in evaluation: {e}")
        return None
    async def multi_turn(self,sample):
        history=[]
        conversavtion={}
        conversavtion["id"]=sample["id"]
        conversavtion["messages"]=[]
        convs=sample["messages"]
        num=len(convs)
        for i in range(num // 2):
            user_conv=convs[i*2]
            assistant_conv=convs[i*2+1]
            response=await self.infer_single(user_conv["content"],assistant_conv["content"],history)
            conversavtion["messages"].extend(response)
            history.extend(response)
        return conversavtion

    async def infer_batch(self, samples, batch_size=5,output_file=""):
        results = []
        if os.path.exists(output_file):
            with open(output_file,'r',encoding="utf-8") as f:
                results=read_from_json(output_file)
            alreadyIdx=[sample["id"] for sample in results]
            samples=[sample for sample in samples if sample["id"] not in alreadyIdx]
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            tasks = [
                self.multi_turn(sample)
                for sample in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for sample, result in zip(batch, batch_results):
                if result:
                    print(sample['id'],len(results))
                    results.append(result)
            with open(output_file,"w",encoding="utf-8") as f:
                json.dump(results,f,ensure_ascii=False,indent=2)
        return results


if __name__ == "__main__":
    root="data/"
    filepath=root+"WhoCreateYou.json"
    filename=os.path.basename(filepath)
    savepath=root+f"distill-{filename}.json"
    samples = read_from_json(filepath)
    api_key = "your_api_key"
    base_url = "http://127.0.0.1:11434/v1"
    distiller = Distiller(api_key, base_url,model="deepseek-r1:32b")
    print(len(samples))
    async def main():
        results = await distiller.infer_batch(samples,10,savepath)
    asyncio.run(main())