from __future__ import annotations

import asyncio

from app.config import get_settings
from app.storage.db import open_db, init_db


async def main() -> None:
    s = get_settings()
    conn = await open_db(s.sqlite_path)
    try:
        await init_db(conn)
        print("DB_INIT_OK")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
