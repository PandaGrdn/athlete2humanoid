# Web app: FastAPI backend (server/) + Vite/React frontend (web/). See docs/DEPLOY.md.
PY ?= python3
NPM ?= npm

.PHONY: dev api web install test space

install:
	$(PY) -m pip install -r server/requirements-dev.txt
	cd web && $(NPM) install

# API on :8000 and the UI on :5173 (proxies /api). Ctrl-C stops both.
dev:
	@trap 'kill 0' INT TERM; $(MAKE) --no-print-directory api & $(MAKE) --no-print-directory web & wait

api:
	$(PY) -m uvicorn server.main:app --reload --reload-dir server --port 8000

web:
	cd web && $(NPM) run dev

test:
	$(PY) -m pytest server/tests -q
	$(PY) -m unittest discover -s tests -t .
	cd web && $(NPM) run typecheck && $(NPM) run build

# The Hugging Face Space folder (build/space). CORS=<frontend origin>, ONNX_PY=<python with torch+onnx>
space:
	$(PY) scripts/build_space.py --cors "$(or $(CORS),http://localhost:5173)" $(if $(ONNX_PY),--onnx-python $(ONNX_PY))
