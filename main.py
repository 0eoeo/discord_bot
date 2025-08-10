import os
import asyncio
import discord
from discord.ext import commands
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat.chat_models import GigaChat
from fastapi import FastAPI
import uvicorn
import base64
from pathlib import Path
import re

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", '')
PORT = int(os.getenv("PORT", "8080"))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

llm = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    model="GigaChat-Pro",
    timeout=6000,
    verify_ssl_certs=False,
)
llm = llm.bind_tools(tools=[], tool_choice="auto")

system_message = SystemMessage(
    content="Ты эмпатичный бот-психолог, который помогает пользователю решить его проблемы."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message.content),
        MessagesPlaceholder("history", optional=True),
        ("user", "{user_input}"),
    ]
)

conversations = {}

def generate_image_and_description(llm, user_id, prompt_text):
    chain = prompt | llm
    response = chain.invoke({"user_input": prompt_text})
    image_uuid = response.additional_kwargs.get("image_uuid")
    description = response.additional_kwargs.get("postfix_message", "")
    if not image_uuid:
        return None, response.content
    img_file = llm.get_file(image_uuid)
    return img_file.content, description

@bot.event
async def on_ready():
    print(f"Бот запущен как {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or message.channel.id != CHANNEL_ID:
        return

    user_id = message.author.id
    user_msg = message.content.strip()

    if user_id not in conversations:
        conversations[user_id] = [system_message]

    conversations[user_id].append(HumanMessage(content=user_msg))
    loop = asyncio.get_event_loop()

    try:
        if "нарисуй" in user_msg.lower():
            image_b64, description = await loop.run_in_executor(
                None, generate_image_and_description, llm, user_id, user_msg
            )
            if image_b64:
                img_bytes = base64.b64decode(image_b64)
                tmp_path = Path(f"tmp_{user_id}.png")
                tmp_path.write_bytes(img_bytes)
                await message.channel.send(
                    file=discord.File(str(tmp_path)),
                    content=description or "Вот твоя картинка!"
                )
                tmp_path.unlink()
            else:
                await message.channel.send(description or "Не удалось сгенерировать картинку.")
        else:
            def get_text_response():
                return llm.invoke(conversations[user_id])
            response = await loop.run_in_executor(None, get_text_response)
            conversations[user_id].append(response)

            # Убираем <img src="UUID" fuse="true"/>
            clean_text = re.sub(r'<img src=".*?" fuse="true"\s*/?>', '', response.content).strip()

            await message.channel.send(clean_text)

    except Exception as e:
        await message.channel.send(f"Ошибка: {e}")

# Создаем веб-сервер для указания порта
app = FastAPI()

@app.get("/")
def root():
    return {"status": "Bot is running"}

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(bot.start(DISCORD_TOKEN))
    uvicorn.run(app, host="0.0.0.0", port=PORT)

