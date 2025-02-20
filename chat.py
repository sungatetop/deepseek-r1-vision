import time
import base64
from io import BytesIO
import os
from openai import AsyncOpenAI
from PIL import Image
import chainlit as cl
# 定义保存图片的目录
SAVE_DIR = "received_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
client = AsyncOpenAI(
    api_key="sk-",
    base_url="http://localhost:8000/v1"
)
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            )
        ]
@cl.on_message
async def on_message(message: cl.Message):
    start = time.time()
    history= cl.chat_context.to_openai()[:-1]
    # 处理所有图片文件
    images = []
    for file in message.elements or []:
        if "image" in file.mime and file.path is not None:
            try:
                img = Image.open(file.path)
                images.append(img)

                # 生成保存图片的文件名，使用时间戳确保唯一性
                timestamp = int(time.time() * 1000)
                save_path = os.path.join(SAVE_DIR, f"{timestamp}.png")
                img.save(save_path, format="PNG")
                print(f"Image saved to {save_path}")
            except Exception as e:
                print(f"Error opening or saving image: {e}")

    # 构造消息内容
    user_content = []
    
    # 添加文本内容
    if message.content.strip():
        content=message.content
        if images:
            content="<image>"*len(images)+message.content
        user_content.append({"type": "text", "text": content})
    
    # 添加图片内容
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    # 构造完整的消息结构
    messages = [
        {"role": "system", "content": ""},
        *history,
        {"role": "user", "content": user_content}
    ]
    print(messages)
    # 创建API请求
    stream = await client.chat.completions.create(
        model="/modelpath",# if use vllm directly
        messages=messages,
        #max_tokens=1024*64,
        stream=True
    )

    thinking = False
    async with cl.Step(name="Thinking") as thinking_step:
        final_answer = cl.Message(content="")
        message=""
        async for chunk in stream:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            message+=delta.content
            # 处理特殊标记
            if "<think>" in delta.content:
                thinking = True
                continue
                
            if "</think>" in delta.content:
                thinking = False
                thought_for = round(time.time() - start)
                thinking_step.name = f"思考...用时{thought_for}s"
                await thinking_step.update()
                continue
            
            # 分流处理思考过程和最终答案
            if thinking:
                await thinking_step.stream_token(delta.content)
            else:
                await final_answer.stream_token(delta.content)
        print(message)
    await final_answer.send()