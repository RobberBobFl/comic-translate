"""Local translation memory store for future RAG / model training.

This is intentionally a *separate* global SQLite database living under the
application user-data directory (``get_user_data_dir()``). It accumulates
translation records across all projects and is never part of a ``.ctpr``
project file, so user corrections are private and excluded from any project
share / git commit.

Scope of this module (per task): only reliable *data collection*. No RAG,
embeddings, similarity search, or training logic lives here.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from typing import Iterable, Optional

from modules.utils.paths import get_user_data_dir

logger = logging.getLogger(__name__)

# HTML document signatures. A stray ``<!DOCTYPE …>`` (as emitted by Qt's
# QTextDocument.toHtml()) is the strongest signal; the rest catch markup that
# should never appear in a translation. Deliberately NOT a bare ``<[^>]+>``
# check so ordinary translations like "a < b" are not rejected.
_HTML_MARKUP_RE = re.compile(
    r"<!\s*doctype|<\s*html|</\s*html>|<\s*head|<\s*body|"
    r"<\s*p\s|<\s*div|<\s*span|<\s*font|<\s*table",
    re.IGNORECASE,
)


def _is_html_markup(text: str) -> bool:
    """True if ``text`` looks like an HTML document / markup rather than plain
    translation text."""
    if not text:
        return False
    return bool(_HTML_MARKUP_RE.search(text))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_memory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       TEXT NOT NULL,
    block_id      TEXT NOT NULL UNIQUE,
    source        TEXT NOT NULL DEFAULT '',
    model_output  TEXT NOT NULL DEFAULT '',
    final_output  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
)
"""

_DB_FILENAME = "translation_memory.db"


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> str:
    base = get_user_data_dir("ComicTranslate")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, _DB_FILENAME)


class TranslationMemoryStore:
    """Persists translation records keyed by a stable per-block ``block_id``."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _db_path()
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    # -- writes -------------------------------------------------------------

    def record_initial(self, page_id: str, block_id: str, source: str, model_output: str) -> None:
        """Capture the model's first answer for a block.

        Skips blank/unsuccessful ``source`` or ``model_output`` so the dataset
        never collects noise. On re-translation the model output is refreshed
        but an existing user ``final_output`` correction is never overwritten.
        """
        if not block_id:
            return
        if not (source or "").strip() or not (model_output or "").strip():
            return
        logger.debug(
            "TM record_initial: page=%s block=%s source=%r model_output=%r",
            page_id, block_id, source, model_output,
        )
        ts = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO translation_memory
                    (page_id, block_id, source, model_output, final_output, created_at, updated_at)
                VALUES (?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    page_id = excluded.page_id,
                    source = excluded.source,
                    model_output = excluded.model_output,
                    updated_at = excluded.updated_at
                """,
                (page_id, block_id, source, model_output, ts, ts),
            )
            self._conn.commit()

    def save_correction(self, page_id: str, block_id: str, final_output: str,
                        source: Optional[str] = None, model_output: Optional[str] = None) -> None:
        """Record (upsert) the user's explicit correction for a block.

        One record per block: repeated presses update ``final_output`` in place
        instead of creating duplicates. Existing ``model_output``/``source`` are
        preserved if the record already exists.

        Safety net: ``final_output`` must be ordinary translation text. If it
        looks like HTML/markup (e.g. a stray ``<!DOCTYPE …>`` document), the
        write is refused so corrupted data never reaches the memory.
        """
        if not block_id:
            return
        if _is_html_markup(final_output or ""):
            logger.error(
                "Refusing to store non-translation final_output for block %s "
                "(value looks like HTML/markup).",
                block_id,
            )
            return
        logger.debug(
            "TM save_correction: page=%s block=%s final_output=%r",
            page_id, block_id, final_output,
        )
        ts = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO translation_memory
                    (page_id, block_id, source, model_output, final_output, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    page_id = excluded.page_id,
                    final_output = excluded.final_output,
                    updated_at = excluded.updated_at
                """,
                (
                    page_id,
                    block_id,
                    source or "",
                    model_output or "",
                    final_output,
                    ts,
                    ts,
                ),
            )
            self._conn.commit()

    def capture_translated_blocks(self, page_id: str, blocks: Iterable) -> None:
        """Helper to capture a list of translated blocks after a successful run.

        Each block must expose ``.block_id`` (via ``ensure_block_id``),
        ``.text`` (source) and ``.translation`` (model output). Blank entries
        are skipped by ``record_initial``.
        """
        from modules.utils.textblock import ensure_block_id

        for blk in blocks:
            block_id = ensure_block_id(blk)
            self.record_initial(page_id, block_id, getattr(blk, "text", ""), getattr(blk, "translation", ""))

    # -- reads / export ----------------------------------------------------

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM translation_memory").fetchone()
            return int(row[0]) if row else 0

    def get(self, block_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT page_id, block_id, source, model_output, final_output, created_at, updated_at "
                "FROM translation_memory WHERE block_id = ?",
                (block_id,),
            ).fetchone()
            if row is None:
                return None
            cols = ("page_id", "block_id", "source", "model_output", "final_output", "created_at", "updated_at")
            return dict(zip(cols, row))

    def export_jsonl(self, path: str) -> int:
        """Write all records as JSONL with source/model_output/final_output.

        Returns the number of written lines.
        """
        import json

        written = 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT source, model_output, final_output FROM translation_memory"
            ).fetchall()
        with open(path, "w", encoding="utf-8") as fh:
            for source, model_output, final_output in rows:
                fh.write(json.dumps(
                    {
                        "source": source or "",
                        "model_output": model_output or "",
                        "final_output": final_output or "",
                    },
                    ensure_ascii=False,
                ))
                fh.write("\n")
                written += 1
        return written

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_store_singleton: Optional[TranslationMemoryStore] = None
_store_lock = threading.Lock()


def get_translation_memory() -> TranslationMemoryStore:
    """Process-wide singleton backed by the global user-data DB."""
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                _store_singleton = TranslationMemoryStore()
    return _store_singleton
