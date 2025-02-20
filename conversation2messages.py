import json
import os
from tqdm import tqdm
filepath="data/cog_10.json"
filename=os.path.basename(filepath).split(".")[0]
with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)
with_image_data=[]
only_text_data=[]
count=0
for i, d in enumerate(tqdm(data)):
    id=d["id"]
    conversations=d["conversations"]
    c={}
    c["messages"]=[]
    c["id"]=id
    if "image" in d.keys():
        image=d["image"]
        c["images"]=["images/"+image]
        
        for idx,conv in enumerate(conversations):
            cfrom=conv["from"]
            cvalue=conv["value"]
            if idx==0:
                cvalue="<image>"+cvalue
            
            if cfrom=="human":
                role="user"
            if cfrom=="gpt":
                role="assistant"
            c["messages"].append({"content":cvalue,"role":role})          
        with_image_data.append(c)
    else:
        for idx,conv in enumerate(conversations):
            cfrom=conv["from"]
            cvalue=conv["value"]           
            if cfrom=="human":
                role="user"
            if cfrom=="gpt":
                role="assistant"
            c["messages"].append({"content":cvalue,"role":role})   
        only_text_data.append(c)
print("data only_text_data items:",len(only_text_data))
print("data with_image_data items:",len(with_image_data))
if len(only_text_data)>0:
    with open(f"data/{filename}.json",'w',encoding="utf-8") as f:
        json.dump(only_text_data,f,ensure_ascii=False,indent=2)
if len(with_image_data)>0:
    with open(f"data/{filename}.json",'w',encoding="utf-8") as f:
        json.dump(with_image_data,f,ensure_ascii=False,indent=2)
