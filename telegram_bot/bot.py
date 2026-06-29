"""
Run after:
  - step 1's ingest.py has been run
  - GROQ_API_KEY is set in your .env file
  - TELEGRAM_BOT_TOKEN is set in your .env file (get one from @BotFather
    on Telegram - message it, /newbot, follow the prompts, it gives you
    a token in under a minute, no card/verification needed)

Usage:
    python telegram_bot/bot.py

Then message your bot on Telegram from any device.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from agent.graph import build_graph
from agent.llm_client import GroqClient
from agent.retriever import KnowledgeRetriever
from agent.product_loader import load_product_searcher_or_none
from telegram_bot.handler import handle_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text

    result = handle_message(context.bot_data["graph"], chat_id, text)

    reply = result["response"]
    if result["escalate"]:
        reply += "\n\n🔴 _Escalated to a human agent_"
    elif result.get("needs_clarification"):
        reply += "\n\n🟡 _Clarifying_"

    await update.message.reply_text(reply, parse_mode="Markdown")


def main():
    groq_key = os.environ.get("GROQ_API_KEY")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")

    if not groq_key:
        print("GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)
    if not bot_token:
        print("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")
        print("Get one by messaging @BotFather on Telegram and running /newbot")
        sys.exit(1)

    llm = GroqClient()
    retriever = KnowledgeRetriever()
    product_searcher = load_product_searcher_or_none()
    graph = build_graph(llm, retriever, product_searcher)

    app = Application.builder().token(bot_token).build()
    app.bot_data["graph"] = graph
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("Rafiq Telegram bot is running. Message it on Telegram now.")
    app.run_polling()


if __name__ == "__main__":
    main()
