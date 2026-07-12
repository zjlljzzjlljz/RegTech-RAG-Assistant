from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, RowMapping

from src.storage.transaction_db import TransactionLog


class PostgresTransactionRepository:
    def __init__(self, database_url: str) -> None:
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True, future=True)

    def insert(self, log: TransactionLog) -> int:
        statement = text(
            """
            INSERT INTO transaction_logs (
                request_id, query, answer, claims_json, feedback, error_message,
                prompt_tokens, completion_tokens, total_tokens, iterations,
                latency_ms, audit_status, model_versions_json, evidence_ids_json, created_at
            ) VALUES (
                CAST(:request_id AS UUID), :query, :answer, CAST(:claims_json AS JSONB),
                :feedback, :error_message, :prompt_tokens, :completion_tokens,
                :total_tokens, :iterations, :latency_ms, :audit_status,
                CAST(:model_versions_json AS JSONB), CAST(:evidence_ids_json AS JSONB),
                CAST(:created_at AS TIMESTAMPTZ)
            ) RETURNING id
            """
        )
        with self._engine.begin() as connection:
            row_id = connection.execute(statement, log.__dict__).scalar_one()
        return int(row_id)

    def update_feedback(self, log_id: int, feedback: str | None) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text("UPDATE transaction_logs SET feedback=:feedback WHERE id=:id"),
                {"feedback": feedback, "id": log_id},
            )

    def get_recent(self, limit: int = 50) -> list[TransactionLog]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text("SELECT * FROM transaction_logs ORDER BY created_at DESC LIMIT :limit"),
                {"limit": limit},
            ).mappings()
            return [self._from_mapping(row) for row in rows]

    def count(self) -> int:
        with self._engine.connect() as connection:
            return int(connection.execute(text("SELECT COUNT(*) FROM transaction_logs")).scalar_one())

    def _from_mapping(self, row: RowMapping) -> TransactionLog:
        def json_text(value: Any) -> str:
            import json

            return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value

        return TransactionLog(
            id=int(row["id"]),
            request_id=str(row["request_id"]) if row["request_id"] else None,
            query=str(row["query"]),
            answer=str(row["answer"]),
            claims_json=json_text(row["claims_json"]),
            feedback=row["feedback"],
            error_message=row["error_message"],
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            total_tokens=int(row["total_tokens"]),
            iterations=int(row["iterations"]),
            latency_ms=float(row["latency_ms"]),
            audit_status=str(row["audit_status"]),
            model_versions_json=json_text(row["model_versions_json"]),
            evidence_ids_json=json_text(row["evidence_ids_json"]),
            created_at=row["created_at"].isoformat(),
        )


__all__ = ["PostgresTransactionRepository"]
