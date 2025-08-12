import os
import re
import base64
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

import discord
from discord.ext import commands
from fastapi import FastAPI
import uvicorn
import yt_dlp
import tempfile

# GigaChat / Langchain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat.chat_models import GigaChat

# ====== Конфигурация ======
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0") or 0)
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
PORT = int(os.getenv("PORT", "8080"))

if not DISCORD_TOKEN:
    raise SystemExit("Нужно указать DISCORD_TOKEN в окружении")
if CHANNEL_ID == 0:
    raise SystemExit("Нужно указать CHANNEL_ID в окружении")
if not GIGACHAT_CREDENTIALS:
    raise SystemExit("Нужно указать GIGACHAT_CREDENTIALS в окружении")

# ====== Discord bot ======
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ====== GigaChat LLM setup ======
llm = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
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

conversations: dict[int, list] = {}  # история по пользователям
IMG_TAG_REGEX = re.compile(r'<img\s+src="([^"]+)"(?:\s+fuse="true")?\s*/?>', flags=re.IGNORECASE)

# ====== GigaChat image generation helper (sync) ======

def generate_image_and_description_sync(prompt_text: str):
    chain = prompt | llm
    response = chain.invoke({"user_input": prompt_text})
    image_uuid = response.additional_kwargs.get("image_uuid")
    description = response.additional_kwargs.get("postfix_message", "") or response.content
    description = IMG_TAG_REGEX.sub("", description).strip()
    if not image_uuid:
        return None, description
    img_file = llm.get_file(image_uuid)
    return img_file.content, description

# ====== Discord events & music commands ======

@bot.event
async def on_ready():
    print(f"[bot] Logged in as {bot.user} (id: {bot.user.id})")

@bot.command(name="leave")
async def cmd_leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Отключился.")
    else:
        await ctx.send("Я никуда не подключён.")

@bot.command(name="play", aliases=["p"])
async def cmd_play(ctx, *, query: str):
    # Проверяем голосовой канал пользователя
    if ctx.author.voice is None:
        await ctx.send("Нужно быть в голосовом канале.")
        return

    voice = ctx.voice_client

    # Если бота нет в голосовом канале — подключаемся туда, где пользователь
    if voice is None:
        voice = await ctx.author.voice.channel.connect()
    else:
        # Если бот в другом канале, переключаемся
        if voice.channel != ctx.author.voice.channel:
            await voice.move_to(ctx.author.voice.channel)

    await ctx.send(f'Ищу и скачиваю с YouTube: "{query}"...')

    loop = asyncio.get_running_loop()

    def download_audio_from_youtube(search_query):
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'outtmpl': tempfile.gettempdir() + '/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'socket_timeout': 60,
            'readtimeout': 60,
            'retries': 10,
            'http_chunk_size': 10485760,
            'nooverwrites': True,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch:{search_query}", download=True)['entries'][0]
            return ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp3', info.get('title', 'Unknown Title')
    try:
        audio_path, title = await loop.run_in_executor(None, download_audio_from_youtube, query)
        source = discord.FFmpegPCMAudio(audio_path)

        def after_playback(err):
            try:
                os.remove(audio_path)
            except Exception:
                pass
            if err:
                print(f"Playback error: {err}")
                coro = ctx.send(f"Ошибка воспроизведения: {err}")
                asyncio.run_coroutine_threadsafe(coro, bot.loop)

        if voice.is_playing() or voice.is_paused():
            voice.stop()
            await asyncio.sleep(0.3)

        voice.play(source, after=after_playback)
        await ctx.send(f"Проигрываю: {title}")

    except Exception as e:
        await ctx.send(f"Ошибка: {e}")

@bot.command(name="stop")
async def cmd_stop(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("Остановлено.")
    else:
        await ctx.send("Нечего останавливать.")

# ====== GigaChat text / image handling in on_message ======

@bot.event
async def on_message(message: discord.Message):
    print(f"Got message from {message.author} (id={message.author.id}), webhook_id={message.webhook_id}")

    if message.author.id == bot.user.id or message.webhook_id is not None:
        return

    # Важно обрабатывать команды *после* проверки (иначе может быть рекурсия)
    await bot.process_commands(message)

    if message.channel.id != CHANNEL_ID:
        return

    user_id = message.author.id
    text = message.content.strip()
    if not text:
        return

    if user_id not in conversations:
        conversations[user_id] = [system_message]
    conversations[user_id].append(HumanMessage(content=text))

    loop = asyncio.get_running_loop()

    try:
        if "нарисуй" in text.lower():
            image_b64, description = await loop.run_in_executor(None, generate_image_and_description_sync, text)
            if image_b64:
                img_bytes = base64.b64decode(image_b64)
                tmp = NamedTemporaryFile(delete=False, suffix=".png")
                tmp_path = tmp.name
                tmp.close()
                Path(tmp_path).write_bytes(img_bytes)
                if description:
                    await message.channel.send(content=description, file=discord.File(tmp_path))
                else:
                    await message.channel.send(file=discord.File(tmp_path))
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
                return
            else:
                await message.channel.send(description or "Не удалось сгенерировать картинку.")
                return

        def _invoke(history):
            return llm.invoke(history)

        response = await loop.run_in_executor(None, _invoke, conversations[user_id])
        conversations[user_id].append(response)
        clean_text = IMG_TAG_REGEX.sub("", response.content).strip()
        MAX = 4000
        if len(clean_text) <= MAX:
            await message.channel.send(clean_text)
        else:
            for i in range(0, len(clean_text), MAX):
                await message.channel.send(clean_text[i:i+MAX])

    except Exception as e:
        await message.channel.send(f"Ошибка: {e}")

# ====== FastAPI server ======
app = FastAPI()

@app.get("/")
async def root():
    bot_name = str(bot.user) if bot.user else None
    return {"status": "ok", "bot": bot_name}

# ====== Runner: start bot + server в одном event loop ======

async def start_bot():
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
