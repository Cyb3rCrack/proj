"""SQLite-backed conversation storage for chat sessions."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict


class ConversationStore:
    """Simple conversation store using SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()

        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT messages FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return []
            try:
                messages = json.loads(row[0])
                if isinstance(messages, list):
                    return messages
            except Exception:
                return []
        return []

    def save_messages(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        payload = json.dumps(messages, ensure_ascii=True)
        updated_at = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, messages, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    messages = excluded.messages,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, payload, updated_at),
            )
            conn.commit()

    def append_message(self, conversation_id: str, role: str, content: str) -> List[Dict[str, str]]:
        messages = self.get_messages(conversation_id)
        messages.append({"role": str(role), "content": str(content)})
        self.save_messages(conversation_id, messages)
        return messages
