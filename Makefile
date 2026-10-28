# Web app: FastAPI backend (server/) + Vite/React frontend (web/). See docs/DEPLOY.md.
PY ?= python3
NPM ?= npm

.PHONY: dev api web install test lint audit lock space deploy

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

lint:
	uvx ruff check .

# Non-blocking in CI; here it just reports
audit:
	-uvx pip-audit -r server/requirements.lock --require-hashes --disable-pip
	-uvx pip-audit -r server/requirements-live.lock --require-hashes --disable-pip
	-cd web && $(NPM) audit

# Re-pin the Space's dependencies (hashes for linux x86_64, Python 3.12, the base image's) after editing
# server/requirements.txt or server/requirements-live.txt. The live lock takes torch from the PyTorch CPU index
# and keeps the packages both locks share at the server lock's versions (both are installed with --no-deps).
LOCK_FLAGS = --generate-hashes --python-version 3.12 --python-platform x86_64-manylinux_2_28
lock:
	uv pip compile server/requirements.txt $(LOCK_FLAGS) -o server/requirements.lock
	uv pip compile server/requirements-live.txt $(LOCK_FLAGS) --torch-backend cpu \
	  -c server/requirements.lock --excludes server/requirements-live-excludes.txt -o server/requirements-live.lock

# The Hugging Face Space folder (build/space). CORS=<frontend origin>, ONNX_PY=<python with torch+onnx>
space:
	$(PY) scripts/build_space.py --cors "$(or $(CORS),http://localhost:5173)" $(if $(ONNX_PY),--onnx-python $(ONNX_PY))

# Space + Pages in one go, after `hf auth login` and `npx wrangler login` (scripts/deploy.sh). HF_USER=<you> optional.
deploy:
	bash scripts/deploy.sh $(HF_USER)
