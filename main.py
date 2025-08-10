import os
import asyncio
import re
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile

import discord
from discord.ext import commands
from fastapi import FastAPI
import uvicorn

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat.chat_models import GigaChat

# ==== Конфигурация ====
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
PORT = int(os.getenv("PORT", "8080"))

if not DISCORD_TOKEN:
    raise SystemExit("Нужно указать DISCORD_TOKEN в окружении")
if CHANNEL_ID == 0:
    raise SystemExit("Нужно указать CHANNEL_ID в окружении")
if not GIGACHAT_CREDENTIALS:
    raise SystemExit("Нужно указать GIGACHAT_CREDENTIALS в окружении")

# ==== Discord ====
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ==== GigaChat LLM ====
llm = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    model="GigaChat-Pro",
    timeout=6000,
    verify_ssl_certs=False,
)
llm = llm.bind_tools(tools=[], tool_choice="auto")

SYSTEM_PROMPT_TEXT = "Ты дерзкая богиня луны."

system_message = SystemMessage(content=SYSTEM_PROMPT_TEXT)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEXT),
        MessagesPlaceholder("history", optional=True),
        ("user", "{user_input}"),
    ]
)

# История переписки по каждому пользователю (ограничиваем длину истории)
conversations: dict[int, list] = {}
MAX_HISTORY = 12  # количество элементов history (можно настроить)

IMG_TAG_REGEX = re.compile(r'<img\s+src="([^"]+)"(?:\s+fuse="true")?\s*/?>', flags=re.IGNORECASE)


def generate_image_and_description(prompt_text: str):
    """
    Синхронная функция, вызываемая в executor.
    Возвращает (base64_image_str_or_None, description_text).
    """
    chain = prompt | llm
    response = chain.invoke({"user_input": prompt_text})

    image_uuid = response.additional_kwargs.get("image_uuid")
    description = response.additional_kwargs.get("postfix_message", "") or response.content

    # Убираем html-тег <img ... /> из описания
    description = IMG_TAG_REGEX.sub("", description).strip()

    if not image_uuid:
        return None, description

    img_file = llm.get_file(image_uuid)  # объект с атрибутом .content (base64)
    return img_file.content, description


@bot.event
async def on_ready():
    print(f"[bot] Logged in as {bot.user} (id: {bot.user.id})")


@bot.event
async def on_message(message: discord.Message):
    # базовые фильтры
    if message.author.bot:
        return
    if message.channel.id != CHANNEL_ID:
        return

    user_id = message.author.id
    user_msg = message.content.strip()
    if not user_msg:
        return

    # prepare history
    if user_id not in conversations:
        conversations[user_id] = [system_message]
    conversations[user_id].append(HumanMessage(content=user_msg))

    # trim history to reasonable size
    if len(conversations[user_id]) > MAX_HISTORY:
        # keep system message + last (MAX_HISTORY-1) user/assistant messages
        conversations[user_id] = [conversations[user_id][0]] + conversations[user_id][- (MAX_HISTORY - 1):]

    try:
        loop = asyncio.get_running_loop()

        if "нарисуй" in user_msg.lower():  # триггер для генерации картинки
            # Запускаем синхронную генерацию в executor
            image_b64, description = await loop.run_in_executor(
                None, generate_image_and_description, user_msg
            )

            if image_b64:
                # Декодируем base64 в байты (безопасно)
                try:
                    img_bytes = base64.b64decode(image_b64)
                except Exception as exc:
                    await message.channel.send(f"Ошибка декодирования изображения: {exc}")
                    return

                tmp_file = NamedTemporaryFile(delete=False, suffix=".png")
                tmp_path = Path(tmp_file.name)
                tmp_file.close()
                try:
                    tmp_path.write_bytes(img_bytes)
                    # Отправляем картинку и описание (если есть)
                    if description:
                        await message.channel.send(content=description, file=discord.File(str(tmp_path)))
                    else:
                        await message.channel.send(file=discord.File(str(tmp_path)))
                finally:
                    # обязательно удаляем временный файл
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
            else:
                # нет картинки — отправляем текст-описание
                await message.channel.send(description or "Не удалось сгенерировать картинку.")
        else:
            # Текстовый ответ (invoke с историей) — запуск в executor
            def get_text_response():
                return llm.invoke(conversations[user_id])

            response = await loop.run_in_executor(None, get_text_response)

            # Удаляем возможные img-теги из текста перед отправкой
            clean_text = IMG_TAG_REGEX.sub("", response.content).strip()
            conversations[user_id].append(response)

            # ограничение длины сообщения Discord (4000 символов)
            if len(clean_text) > 4000:
                # разделим на части
                for i in range(0, len(clean_text), 4000):
                    await message.channel.send(clean_text[i:i+4000])
            else:
                await message.channel.send(clean_text)

    except Exception as e:
        # логируем в канал, но без избытка информации
        await message.channel.send(f"Ошибка: {e}")


# ==== FastAPI ====
app = FastAPI()


@app.get("/")
async def root():
    bot_name = str(bot.user) if bot.user else None
    return {"status": "ok", "bot": bot_name}


# ==== Запуск обеих служб в одном event loop ====
async def start_bot():
    # bot.start поддерживает повторные подключения
    await bot.start(DISCORD_TOKEN)


async def start_web():
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, loop="asyncio", log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    await asyncio.gather(start_bot(), start_web())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
