from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import aiosqlite


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def open_db(path: str) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA foreign_keys=ON;")
    return conn


async def init_db(conn: aiosqlite.Connection) -> None:
    await conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,                -- YYYY-MM-DD (UTC)
            created_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL,
            is_closed INTEGER NOT NULL DEFAULT 0,
            user_msg_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user_day ON sessions(user_id, day);

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,               -- 'user' | 'assistant' | 'system'
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """
    )
    await conn.commit()


async def ensure_user(conn: aiosqlite.Connection, user_id: int) -> None:
    now = utcnow().isoformat()
    await conn.execute(
        "INSERT OR IGNORE INTO users(user_id, created_at) VALUES(?, ?);",
        (user_id, now),
    )
    await conn.commit()


async def get_active_session_id(
    conn: aiosqlite.Connection,
    user_id: int,
    day: str,
) -> Optional[int]:
    cur = await conn.execute(
        """
        SELECT id
        FROM sessions
        WHERE user_id = ? AND day = ? AND is_closed = 0
        ORDER BY last_active_at DESC
        LIMIT 1;
        """,
        (user_id, day),
    )
    row = await cur.fetchone()
    return int(row["id"]) if row else None
async def count_sessions_today(conn: aiosqlite.Connection, user_id: int, day: str) -> int:
    cur = await conn.execute(
        "SELECT COUNT(*) AS c FROM sessions WHERE user_id = ? AND day = ?;",
        (user_id, day),
    )
    row = await cur.fetchone()
    return int(row["c"])


async def create_session(conn: aiosqlite.Connection, user_id: int, day: str) -> int:
    now = utcnow().isoformat()
    cur = await conn.execute(
        """
        INSERT INTO sessions(user_id, day, created_at, last_active_at, is_closed, user_msg_count)
        VALUES(?, ?, ?, ?, 0, 0);
        """,
        (user_id, day, now, now),
    )
    await conn.commit()
    return int(cur.lastrowid)


async def touch_session(conn: aiosqlite.Connection, session_id: int) -> None:
    now = utcnow().isoformat()
    await conn.execute(
        "UPDATE sessions SET last_active_at = ? WHERE id = ?;",
        (now, session_id),
    )
    await conn.commit()


async def increment_user_msg_count(conn: aiosqlite.Connection, session_id: int) -> int:
    await conn.execute(
        "UPDATE sessions SET user_msg_count = user_msg_count + 1 WHERE id = ?;",
        (session_id,),
    )
    await conn.commit()
    cur = await conn.execute(
        "SELECT user_msg_count FROM sessions WHERE id = ?;",
        (session_id,),
    )
    row = await cur.fetchone()
    return int(row["user_msg_count"])


async def close_session(conn: aiosqlite.Connection, session_id: int) -> None:
    await conn.execute("UPDATE sessions SET is_closed = 1 WHERE id = ?;", (session_id,))
    await conn.commit()


async def add_message(conn: aiosqlite.Connection, session_id: int, role: str, content: str) -> None:
    now = utcnow().isoformat()
    await conn.execute(
        "INSERT INTO messages(session_id, role, content, created_at) VALUES(?, ?, ?, ?);",
        (session_id, role, content, now),
    )
    await conn.commit()


async def get_last_messages(conn: aiosqlite.Connection, session_id: int, limit: int = 20) -> list[dict]:
    cur = await conn.execute(
        """
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (session_id, limit),
    )
    rows = await cur.fetchall()
    # возвращаем в правильном порядке (старые -> новые)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
