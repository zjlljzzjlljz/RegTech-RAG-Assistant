from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransactionLog:
    """Persisted record of a single compliance audit transaction."""

    query: str
    answer: str
    claims_json: str
    feedback: str | None = None
    error_message: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    iterations: int = 0
    latency_ms: float = 0.0
    request_id: str | None = None
    audit_status: str = "pending"
    model_versions_json: str = "{}"
    evidence_ids_json: str = "[]"
    id: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TransactionLog":
        """Reconstruct a TransactionLog from a sqlite3.Row."""
        return cls(
            id=row["id"],
            query=row["query"],
            answer=row["answer"],
            claims_json=row["claims_json"],
            feedback=row["feedback"] if "feedback" in row.keys() else None,
            error_message=row["error_message"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            iterations=row["iterations"],
            latency_ms=row["latency_ms"],
            request_id=row["request_id"] if "request_id" in row.keys() else None,
            audit_status=row["audit_status"] if "audit_status" in row.keys() else "pending",
            model_versions_json=row["model_versions_json"] if "model_versions_json" in row.keys() else "{}",
            evidence_ids_json=row["evidence_ids_json"] if "evidence_ids_json" in row.keys() else "[]",
            created_at=row["created_at"],
        )


class TransactionRepository:
    """Thread-safe SQLite repository for compliance audit transaction logs.

    Designed for Streamlit's multi-threaded runtime:
    - check_same_thread=False on every connection
    - WAL journal mode for concurrent read/write
    - threading.Lock() guards all public methods
    - Each method owns its connection (never shares, never leaks)
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        if db_path is None:
            from config.settings import resolve_project_root

            root = resolve_project_root()
            db_path = root / "data" / "transactions.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection with WAL mode and Row factory."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes idempotently. Called once at construction."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transaction_logs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    query               TEXT    NOT NULL,
                    answer              TEXT    NOT NULL DEFAULT '',
                    claims_json         TEXT    NOT NULL DEFAULT '[]',
                    feedback            TEXT,
                    error_message       TEXT,
                    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
                    completion_tokens   INTEGER NOT NULL DEFAULT 0,
                    total_tokens        INTEGER NOT NULL DEFAULT 0,
                    iterations          INTEGER NOT NULL DEFAULT 0,
                    latency_ms          REAL    NOT NULL DEFAULT 0.0,
                    request_id          TEXT,
                    audit_status        TEXT    NOT NULL DEFAULT 'pending',
                    model_versions_json TEXT    NOT NULL DEFAULT '{}',
                    evidence_ids_json   TEXT    NOT NULL DEFAULT '[]',
                    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            migrations = [
                "ALTER TABLE transaction_logs ADD COLUMN feedback TEXT",
                "ALTER TABLE transaction_logs ADD COLUMN request_id TEXT",
                "ALTER TABLE transaction_logs ADD COLUMN audit_status TEXT NOT NULL DEFAULT 'pending'",
                "ALTER TABLE transaction_logs ADD COLUMN model_versions_json TEXT NOT NULL DEFAULT '{}'",
                "ALTER TABLE transaction_logs ADD COLUMN evidence_ids_json TEXT NOT NULL DEFAULT '[]'",
            ]
            for statement in migrations:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transaction_logs_created_at "
                "ON transaction_logs(created_at DESC)"
            )
            conn.commit()
            logger.info("transaction_logs table ready at %s", self._db_path)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, log: TransactionLog) -> int:
        """Insert a transaction log and return its row ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO transaction_logs
                        (query, answer, claims_json, feedback, error_message,
                         prompt_tokens, completion_tokens, total_tokens,
                         iterations, latency_ms, request_id, audit_status,
                         model_versions_json, evidence_ids_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log.query,
                        log.answer,
                        log.claims_json,
                        log.feedback,
                        log.error_message,
                        log.prompt_tokens,
                        log.completion_tokens,
                        log.total_tokens,
                        log.iterations,
                        log.latency_ms,
                        log.request_id,
                        log.audit_status,
                        log.model_versions_json,
                        log.evidence_ids_json,
                        log.created_at,
                    ),
                )
                conn.commit()
                row_id = cursor.lastrowid
                if row_id is not None:
                    row_id = int(row_id)
                logger.debug("TransactionLog inserted, id=%s", row_id)
                return row_id
            finally:
                conn.close()

    def update_feedback(self, log_id: int, feedback: str | None) -> None:
        """Update feedback for an existing transaction log."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "UPDATE transaction_logs SET feedback = ? WHERE id = ?",
                    (feedback, log_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_recent(self, limit: int = 50) -> list[TransactionLog]:
        """Return the most recent N transaction logs, newest first."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM transaction_logs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [TransactionLog.from_row(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def count(self) -> int:
        """Return the total number of transaction log rows."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT COUNT(*) AS cnt FROM transaction_logs")
                row = cursor.fetchone()
                return int(row["cnt"]) if row else 0
            finally:
                conn.close()
