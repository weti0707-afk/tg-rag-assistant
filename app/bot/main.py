from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ChatMemberStatus
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import CommandStart
from aiogram.types import Message

from app.config import get_settings
from app.llm.ollama import OllamaClient
from app.rag.index import RagIndex
import app.storage.db as db


router = Router()

RAG: RagIndex | None = None
LLM: OllamaClient | None = None

INDEX_DIR = Path("data/index")
STOPWORDS_RU = {
    "это","как","так","что","для","или","и","а","но","на","по","в","во","к","ко","с","со","из","у","о","об",
    "же","ли","не","нет","да","я","мы","вы","он","она","они","оно","мне","вам","нас","вас","их","его","ее",
    "уже","ещё","еще","только","тоже","тут","там","здесь","будет","есть","быть","про","при"
}

def _tokens(s: str) -> set[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", (s or "").lower())
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS_RU}

def has_overlap(query: str, ctx: str, min_shared: int = 1) -> bool:
    q = _tokens(query)
    c = _tokens(ctx)
    return len(q & c) >= min_shared

DEFAULT_NO_INFO = "В материалах нет информации для ответа."


SYSTEM_PROMPT = (
    "Ты ассистент, который отвечает ТОЛЬКО на основе предоставленных фрагментов материалов.\n"
    "Если в фрагментах нет ответа — скажи: «В материалах нет информации для ответа.»\n"
    "Не добавляй знания извне и не выдумывай.\n"
    "Отвечай кратко и по делу.\n"
)

def format_hits(hits: list[dict], max_total_chars: int = 2500) -> str:
    """
    Собирает найденные фрагменты в один текст для LLM.
    Безопасно обрабатывает пустые/битые hit'ы и ограничивает общий размер.
    """
    parts: list[str] = []
    total = 0

    for i, h in enumerate(hits or [], start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue

        meta = h.get("meta") or {}
        src = meta.get("source") or meta.get("file") or meta.get("path") or meta.get("filename")
        page = meta.get("page") if "page" in meta else meta.get("pageno")

        header_bits = []
        if src:
            header_bits.append(str(src))
        if page is not None:
            header_bits.append(f"стр. {page}")

        header = f"[{i}]"
        if header_bits:
            header += " " + " — ".join(header_bits)
        header += "\n"

        block = header + text

        remaining = max_total_chars - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip() + "…"

        parts.append(block)
        total += len(block) + 2

    return "\n\n".join(parts).strip()

def utc_day_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def normalize_channel(ch: str | None) -> str | None:
    if not ch:
        return None
    ch = ch.strip()
    if not ch:
        return None
    # Если в .env остался плейсхолдер — считаем что проверка выключена
    if "your_channel_or_leave_empty" in ch:
        return None
    if not ch.startswith("@"):
        ch = "@" + ch
    return ch


async def is_subscribed(bot: Bot, user_id: int, channel: str) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=channel, user_id=user_id)
        logging.info("SUB_CHECK user=%s channel=%s status=%s", user_id, channel, getattr(member, "status", None))
        return member.status in (
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR,
        )
    except (TelegramBadRequest, TelegramForbiddenError) as e:
        logging.warning("SUB_CHECK_ERROR user=%s channel=%s err=%r", user_id, channel, e)
        return False


def answer_from_hits(hits: list[dict], max_chars: int) -> str:
    if not hits:
        return DEFAULT_NO_INFO

    # Самый надёжный вариант: показываем фрагмент(ы) напрямую (без фантазий LLM)
    txt = (hits[0].get("text") or "").strip()
    if not txt:
        return DEFAULT_NO_INFO

    # Немного «человечности», но всё ещё строго по материалам
    ans = f"По материалам:\n{txt}"

    if len(ans) > max_chars:
        ans = ans[:max_chars].rstrip() + "…"
    return ans


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not message.from_user:
        return

    s = get_settings()
    channel = normalize_channel(s.required_channel)

    # Проверка подписки
    if channel:
        ok = await is_subscribed(message.bot, message.from_user.id, channel)
        if not ok:
            await message.answer(
                f"Чтобы пользоваться ботом, подпишитесь на канал {channel} и нажмите /start ещё раз."
            )
            return

    user_id = message.from_user.id
    day = utc_day_str()

    conn = await db.open_db(s.sqlite_path)
    try:
        await db.init_db(conn)
        await db.ensure_user(conn, user_id)

        used = await db.count_sessions_today(conn, user_id, day)
        active_id = await db.get_active_session_id(conn, user_id, day)

        if active_id is None and used >= s.daily_sessions_limit:
            await message.answer(
                "Лимит новых сессий на сегодня исчерпан.\n"
                "Попробуйте завтра."
            )
            return

        if active_id is None:
            session_id = await db.create_session(conn, user_id, day)
        else:
            session_id = active_id

        await db.touch_session(conn, session_id)

        await message.answer(
            "Готово ✅\n"
            "Задайте вопрос по загруженным материалам.\n"
            "Если в материалах нет ответа — я так и скажу."
        )
    finally:
        await conn.close()


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    if not message.from_user:
        return

    s = get_settings()

    # Проверка подписки на канал (если задан)
    if s.required_channel and s.required_channel.strip():
        member_ok = await is_subscribed(message.bot, message.from_user.id, s.required_channel.strip())
        logging.info("SUB_CHECK user=%s channel=%s ok=%s", message.from_user.id, s.required_channel, member_ok)
        if not member_ok:
            await message.answer(
                f"Чтобы пользоваться ботом, подпишитесь на канал {s.required_channel} и нажмите /start ещё раз."
            )
            return

    text = (message.text or "").strip()
    if not text:
        return

    if len(text) > s.max_user_message_chars:
        await message.answer(f"Слишком длинное сообщение. Максимум {s.max_user_message_chars} символов.")
        return

    global RAG, LLM
    if RAG is None:
        await message.answer("RAG индекс не инициализирован. Попробуйте позже.")
        return

    user_id = message.from_user.id
    day = utc_day_str()

    conn = await db.open_db(s.sqlite_path)
    try:
        await db.init_db(conn)
        await db.ensure_user(conn, user_id)

        session_id = await db.get_active_session_id(conn, user_id, day)
        if session_id is None:
            await message.answer("Сначала начните сессию командой /start.")
            return

        n = await db.increment_user_msg_count(conn, session_id)
        if n > s.session_max_followups:
            await db.close_session(conn, session_id)
            await message.answer("Лимит уточнений в этой сессии исчерпан. Начните новую: /start")
            return

        await db.add_message(conn, session_id, "user", text)

        hits = RAG.search(text, top_k=s.rag_top_k)
        ctx = format_hits(hits, max_total_chars=2500)

        if not ctx:
            answer = "В материалах нет информации для ответа."
            await message.answer(answer)
            await db.add_message(conn, session_id, "assistant", answer)
            return

        # Пробуем LLM (строго по фрагментам). Если LLM упала — отдаём первый фрагмент.
        answer = ""
        if LLM is not None:
            system = SYSTEM_PROMPT + "\n\nФРАГМЕНТЫ МАТЕРИАЛОВ:\n" + ctx
            user_prompt = f"Вопрос пользователя:\n{text}\n\nОтветь строго по фрагментам материалов."
            try:
                answer = (await LLM.chat(system, user_prompt)).strip()
            except Exception:
                logging.exception("LLM_FAIL user=%s", user_id)
                answer = ""

        if not answer:
            # Фолбэк: первый фрагмент из поиска
            answer = (hits[0].get("text") or "").strip() or "В материалах нет информации для ответа."

        if len(answer) > s.max_answer_chars:
            answer = answer[: s.max_answer_chars].rstrip() + "…"

        await message.answer(answer)
        await db.add_message(conn, session_id, "assistant", answer)
        await db.touch_session(conn, session_id)

    finally:
        await conn.close()
async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    s = get_settings()

    # Проверяем RAG индекс
    if not (INDEX_DIR / "index.faiss").exists():
        raise SystemExit(
            f"RAG индекс не найден в {INDEX_DIR}. Сначала собери индекс: python -m app.rag.build_index"
        )

    global RAG, LLM
    RAG = RagIndex.load(INDEX_DIR)

    # LLM клиент (Ollama). Если Ollama не поднята — бот всё равно будет отвечать цитатой.
    try:
        LLM = OllamaClient(base_url=s.ollama_base_url, model=s.ollama_model)
    except Exception:
        logging.exception("LLM_INIT_ERROR")
        LLM = None

    bot = Bot(token=s.bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

