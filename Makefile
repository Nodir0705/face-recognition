# Convenience targets. Run from repo root.
#
# All Python targets assume `.venv` exists. Run `make install` once first.

PY := .venv/bin/python
PIP := .venv/bin/pip
PYTHONPATH := .

.PHONY: help install install-dev run sync test test-slow clean

help:
	@echo "Targets:"
	@echo "  install      create .venv with system site-packages (Pi: needs picamera2 from apt)"
	@echo "  install-dev  create .venv WITHOUT system site-packages (laptop / non-Pi dev)"
	@echo "  run          start the main web app (camera + recognition + admin)"
	@echo "  sync         start the Google Sheets sync worker"
	@echo "  test         run the fast test suite"
	@echo "  test-slow    run all tests including ones that load InsightFace"
	@echo "  clean        remove .venv, __pycache__, and SQLite WAL cruft"

install:
	python3 -m venv .venv --system-site-packages
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt

install-dev:
	python3 -m venv .venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	@echo
	@echo "Dev install complete. The Pi camera (picamera2) is NOT available here —"
	@echo "the camera layer will fall back to /dev/video0 via OpenCV (USB webcam)."

run:
	PYTHONPATH=$(PYTHONPATH) $(PY) src/web/app.py

sync:
	PYTHONPATH=$(PYTHONPATH) $(PY) src/sync.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest

test-slow:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest --run-slow

clean:
	rm -rf .venv .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f data/attendance.db-wal data/attendance.db-shm
