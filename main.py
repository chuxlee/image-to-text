import asyncio
import base64
import logging
from pathlib import Path
from io import BytesIO

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv

MODEL_NAME = "qwen/qwen3.5-9b"
LM_STUDIO_URL = "http://localhost:1234/v1"
API_KEY = "not-needed"
load_dotenv()
BOT_TOKEN = os.getenv('TOKEN')

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

client = OpenAI(base_url=LM_STUDIO_URL, api_key=API_KEY)

def encode_image(image_bytes: bytes) -> str:
    if image_bytes[:4] == b'\xff\xd8\xff':
        mime = "image/jpeg"
    elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        mime = "image/webp"
    elif image_bytes[:2] in (b'BM', b'BA'):
        mime = "image/bmp"
    else:
        mime = "image/jpeg"  # fallback

    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64_data}"

async def extract_text_from_image(image_bytes: bytes) -> str:
    base64_image = encode_image(image_bytes)

    def query_api():
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Извлеки весь текст с изображения. "
                                "Объедини строки в осмысленные абзацы: удаляй одиночные переносы строк, "
                                "которые разрывают предложения или идущие подряд слова. "
                                "Сохраняй пустые строки только там, где действительно начинается новый абзац. "
                                "Верни **только** очищенный текст, без каких-либо дополнительных комментариев."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image}
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content

    return await asyncio.to_thread(query_api)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Отправь мне фотографию с текстом, и я распознаю его с помощью нейросети Qwen.\n\n"
        "📎 Можно отправить как сжатое фото, так и файл изображения."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Просто отправь мне картинку (сжатое фото или файл), и я пришлю распознанный текст.\n"
        "Никаких дополнительных команд не нужно."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _process_image(update, context, is_document=False)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if doc.mime_type and doc.mime_type.startswith("image/"):
        await _process_image(update, context, is_document=True)
    else:
        await update.message.reply_text("⚠️ Пожалуйста, отправьте файл изображения (JPEG, PNG, WebP, BMP).")

async def _process_image(update: Update, context: ContextTypes.DEFAULT_TYPE, is_document: bool) -> None:
    try:
        processing_msg = await update.message.reply_text("🔍 Распознаю текст...")

        if is_document:
            file = await update.message.document.get_file()
        else:
            photo = update.message.photo[-1]
            file = await photo.get_file()

        image_bytes = await file.download_as_bytearray()
        image_bytes = bytes(image_bytes)

        text = await extract_text_from_image(image_bytes)

        await processing_msg.delete()

        if text and text.strip():
            await update.message.reply_text(
                f"📄 <b>Распознанный текст:</b>\n\n<pre>{text}</pre>",
                parse_mode="HTML"
            )
        else:
            await update.message.reply_text("🤔 Модель не обнаружила текста на изображении.")

    except Exception as e:
        logger.error(f"Ошибка обработки: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Произошла ошибка: {e}")

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))

    logger.info("Бот запущен. Ожидание сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()