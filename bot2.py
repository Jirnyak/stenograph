# bot_decode.py
import os
import subprocess
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import InputMediaDocument

BOT_TOKEN = "8440416114:AAHTwG1qtC5ZGofvPy_d_VqS0PbyNPcH79s"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Gimme picture!")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        file = await update.message.photo[-1].get_file()
    else:
        file = await update.message.document.get_file()

    input_path = "shifr.png"
    output_path = "reconstructed.wav"

    await file.download_to_drive(input_path)
    await update.message.reply_text("Decoding PNG to audio...")

    subprocess.run(["python3.11", "mpdecode.py"], check=True)

    if os.path.exists(output_path):
        await update.message.reply_document(open(output_path, "rb"))
    else:
        await update.message.reply_text("❌ ERROR: Decoding failed!")

    for f in ["input.mp3", "shifr.png", "reconstructed.wav"]:
        if os.path.exists(f):
            os.remove(f)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))

app.run_polling()
