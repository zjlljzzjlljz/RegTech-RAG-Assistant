from __future__ import annotations

from src.storage.transaction_db import TransactionLog, TransactionRepository


def create_transaction_repository():
    from config.settings import get_settings

    settings = get_settings()
    if settings.storage.database_url:
        from src.storage.postgres_repository import PostgresTransactionRepository

        return PostgresTransactionRepository(settings.storage.database_url)
    return TransactionRepository(settings.storage.sqlite_path)

__all__ = [
    "TransactionLog",
    "TransactionRepository",
    "create_transaction_repository",
]
