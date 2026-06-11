VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
STREAMLIT_ENV := STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_FLAGS := --server.headless=true --browser.gatherUsageStats=false

.PHONY: install build build-llamaindex test app app-v2 clean

install:
	@test -x $(PIP) || (echo "Project virtual environment not found at $(VENV). Please create .venv first with a compatible Python interpreter." && exit 1)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

build:
	$(PYTHON) 1_build_db.py

build-llamaindex:
	$(PYTHON) 4_llamaindex_ingest.py

test:
	$(PYTHON) 2_test_db.py

app:
	$(STREAMLIT_ENV) $(STREAMLIT) run 3_app.py $(STREAMLIT_FLAGS)

app-v2:
	$(STREAMLIT_ENV) $(STREAMLIT) run 8_app.py $(STREAMLIT_FLAGS)

clean:
	rm -rf chroma_db/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned: chroma_db/, __pycache__/, *.pyc"
