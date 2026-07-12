VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
STREAMLIT_ENV := STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_FLAGS := --server.headless=true --browser.gatherUsageStats=false

.PHONY: install test app rebuild-milvus eval-retrieval eval-generation migrate compose-up clean

install:
	@test -x $(PIP) || (echo "Project virtual environment not found at $(VENV). Please create .venv first with a compatible Python interpreter." && exit 1)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

app:
	$(STREAMLIT_ENV) $(STREAMLIT) run app.py $(STREAMLIT_FLAGS)

rebuild-milvus:
	$(PYTHON) -m src.indexing.rebuild_milvus --pdf-dir data/raw_pdfs --drop-existing

eval-retrieval:
	$(PYTHON) -m src.evaluation.eval_retrieval --suite all --fusion rrf --with-rerank

eval-generation:
	EVAL_SUITE=all $(PYTHON) -m src.evaluation.eval_generation

migrate:
	$(VENV)/bin/alembic upgrade head

compose-up:
	docker compose --profile app --profile gpu up -d

clean:
	rm -rf chroma_db/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned: chroma_db/, __pycache__/, *.pyc"
