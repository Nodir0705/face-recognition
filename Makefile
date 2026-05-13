# Convenience targets. Run from repo root.
#
# All Python targets assume `.venv` exists. Run `make install` once first.

PY := .venv/bin/python
PIP := .venv/bin/pip
PYTHONPATH := .

# C++ build settings — override on the command line, e.g.:
#   make cpp-build ONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.1
ONNXRUNTIME_ROOT ?= $(HOME)/onnxruntime-linux-x64-1.18.1
MODELS_DIR       ?= $(HOME)/.insightface/models/buffalo_sc
BENCH_IMAGE      ?= samples/test.jpg
BENCH_ITERS      ?= 200
BENCH_WARMUP     ?= 30
BENCH_THREADS    ?= 2

.PHONY: help install install-dev run sync test test-slow clean \
        cpp-build cpp-clean bench-cpp bench-py run-cpp

help:
	@echo "Python targets:"
	@echo "  install       create .venv with system site-packages (Pi: needs picamera2 from apt)"
	@echo "  install-dev   create .venv WITHOUT system site-packages (laptop / non-Pi dev)"
	@echo "  run           start the main web app (camera + recognition + admin)"
	@echo "  sync          start the Google Sheets sync worker"
	@echo "  test          run the fast test suite"
	@echo "  test-slow     run all tests including ones that load InsightFace"
	@echo "  clean         remove .venv, __pycache__, and SQLite WAL cruft"
	@echo
	@echo "C++ targets (need libopencv-dev, libsqlite3-dev, ONNXRuntime — see cpp/README.md):"
	@echo "  cpp-build     configure + build cpp/build/{bench_cpp,recognize_cpp}"
	@echo "  cpp-clean     rm -rf cpp/build"
	@echo "  bench-cpp     run the C++ benchmark on BENCH_IMAGE (default: $(BENCH_IMAGE))"
	@echo "  bench-py      run the Python benchmark on BENCH_IMAGE"
	@echo "  run-cpp       run the C++ recognition daemon (uses data/attendance.db)"

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

# ---------- C++ ----------

cpp-build:
	cmake -S cpp -B cpp/build \
	      -DONNXRUNTIME_ROOT=$(ONNXRUNTIME_ROOT) -DNATIVE=ON
	cmake --build cpp/build -j

cpp-clean:
	rm -rf cpp/build

bench-cpp: cpp/build/bench_cpp
	./cpp/build/bench_cpp \
	    --model $(MODELS_DIR)/det_500m.onnx \
	    --rec   $(MODELS_DIR)/w600k_mbf.onnx \
	    --image $(BENCH_IMAGE) \
	    --iters $(BENCH_ITERS) --warmup $(BENCH_WARMUP) --threads $(BENCH_THREADS)

bench-py:
	PYTHONPATH=$(PYTHONPATH) $(PY) scripts/bench_python.py \
	    --image $(BENCH_IMAGE) \
	    --iters $(BENCH_ITERS) --warmup $(BENCH_WARMUP) --threads $(BENCH_THREADS) --rec

run-cpp: cpp/build/recognize_cpp
	./cpp/build/recognize_cpp \
	    --models $(MODELS_DIR) \
	    --db     $(PWD)/data/attendance.db \
	    --config $(PWD)/config/config.yaml \
	    --camera 0

cpp/build/bench_cpp cpp/build/recognize_cpp: cpp-build
