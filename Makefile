VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
STREAMLIT_ENV := STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_FLAGS := --server.headless true --browser.gatherUsageStats false

.PHONY: install build test app

install:
	@test -x $(PIP) || (echo "Project virtual environment not found at $(VENV). Please create .venv first with a compatible Python interpreter." && exit 1)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

build:
	$(PYTHON) 1_build_db.py

test:
	$(PYTHON) test_db.py

app:
	$(STREAMLIT_ENV) $(STREAMLIT) run app.py $(STREAMLIT_FLAGS)
