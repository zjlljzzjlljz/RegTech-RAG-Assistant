FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir alembic sqlalchemy "psycopg[binary]"
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
CMD ["alembic", "upgrade", "head"]
