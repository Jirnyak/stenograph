import os
import subprocess
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import InputMediaDocument

BOT_TOKEN = "8449959639:AAH4sASdQP0HAGmFpml2SG2AsHdJrL09oYE"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gimme music!")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for f in ["input.mp3", "shifr.png", "amplitude.png", "phase.png", "reconstructed.wav"]:
        if os.path.exists(f):
            os.remove(f)

    if update.message.audio:
        file = await update.message.audio.get_file()
    elif update.message.document:
        file = await update.message.document.get_file()
    elif update.message.voice:
        file = await update.message.voice.get_file()
    else:
        await update.message.reply_text("❌ Not an audio file or voice!")
        return

    input_path = "input.mp3"
    output_path = "shifr.png"

    await file.download_to_drive(input_path)
    await update.message.reply_text("Encoding...")

    subprocess.run(["python3.11", "mpshifr.py"], check=True)

    media = [
    InputMediaDocument(open("shifr.png", "rb"), caption="encoded audio"),
    InputMediaDocument(open("amplitude.png", "rb"), caption="amplitude data"),
    InputMediaDocument(open("phase.png", "rb"), caption="phase data")
    ]

    if os.path.exists(output_path):
        #await update.message.reply_document(open(output_path, "rb"), filename="encoded.png")
        await update.message.reply_media_group(media)
    else:
        await update.message.reply_text("❌ ERROR: Encoding failed!")

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))

app.add_handler(MessageHandler(filters.Document.ALL | filters.AUDIO | filters.VOICE, handle_audio))

app.run_polling()
