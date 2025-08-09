import base64
import os
import re
from pathlib import Path

import discord
from discord.ext import commands
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat.chat_models import GigaChat

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

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
    content="Ты дерзкая богиня луны."
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
        def get_text_response():
            return llm.invoke(conversations[user_id])

        response = await loop.run_in_executor(None, get_text_response)
        conversations[user_id].append(response)

        content = response.content

        img_match = re.search(r'<img\s+src="([^"]+)"\s+fuse="true"\s*/?>', content)
        if img_match:
            image_uuid = img_match.group(1)

            text_without_img = re.sub(r'<img\s+src="[^"]+"\s+fuse="true"\s*/?>', '', content).strip()

            img_file = llm.get_file(image_uuid)
            img_bytes = base64.b64decode(img_file.content)
            tmp_path = Path(f"tmp_{user_id}.png")
            tmp_path.write_bytes(img_bytes)

            if text_without_img:
                await message.channel.send(content=text_without_img, file=discord.File(str(tmp_path)))
            else:
                await message.channel.send(file=discord.File(str(tmp_path)))

            tmp_path.unlink()
        else:
            await message.channel.send(content)

    except Exception as e:
        await message.channel.send(f"Ошибка: {e}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)