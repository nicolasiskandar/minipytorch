#!/bin/bash
cd "$(dirname "$0")" || exit 1

py_tests='
import sys, os, glob, importlib
sys.path.insert(0, os.getcwd())
# recursive: tests_py/nn/*.py and friends are picked up alongside tests_py/*.py
files = [f for f in sorted(glob.glob("tests_py/**/*.py", recursive=True))
         if os.path.basename(f) != "__init__.py"]
if not files:
    raise SystemExit("no python tests found under tests_py/")
total = 0
ran = []
for f in files:
    mod = f[:-3].replace(os.sep, ".")
    m = importlib.import_module(mod)
    names = sorted(k for k in dir(m) if k.startswith("test_"))
    for k in names:
        getattr(m, k)()
    if names:
        ran.append((mod, len(names)))
    total += len(names)
for mod, n in ran:
    print(f"  {mod}: {n}")
print(f"all python tests passed ({total} in {len(ran)} modules)")
'

require_venv() {
  if [ ! -x .venv/bin/python ]; then
    echo "error: .venv not found - run ./run.sh setup first" >&2
    exit 1
  fi
}

case "$1" in
  setup)
    if ! command -v uv >/dev/null 2>&1; then
      echo "error: uv not found - see https://docs.astral.sh/uv/" >&2
      exit 1
    fi
    if [ ! -x .venv/bin/python ]; then
      uv venv .venv || exit 1
    fi
    uv pip install -p .venv/bin/python -e . || exit 1
    echo "setup complete - try ./run.sh demo"
    ;;
  test)
    cd backend_cxx && make test
    ;;
  test-py)
    require_venv
    .venv/bin/python -c "$py_tests"
    ;;
  test-all)
    require_venv
    (cd backend_cxx && make test) || exit 1
    echo "---"
    .venv/bin/python -c "$py_tests"
    ;;
  demo)
    require_venv
    .venv/bin/python demo.py
    ;;
  build-ext)
    require_venv
    CMAKE=.venv/bin/cmake
    if [ ! -x "$CMAKE" ]; then
      CMAKE=cmake
    fi
    if ! command -v "$CMAKE" >/dev/null 2>&1; then
      echo "error: cmake not found - run: uv pip install -p .venv/bin/python cmake" >&2
      exit 1
    fi
    if ! .venv/bin/python -c "import pybind11" 2>/dev/null; then
      echo "error: pybind11 not found - run: uv pip install -p .venv/bin/python pybind11" >&2
      exit 1
    fi
    pybind11_dir=$(.venv/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())") || exit 1
    "$CMAKE" -S . -B build/ext \
      -Dpybind11_DIR="$pybind11_dir" \
      -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
      -DCMAKE_BUILD_TYPE=Release || exit 1
    "$CMAKE" --build build/ext -j || exit 1
    "$CMAKE" --install build/ext --prefix "$PWD" || exit 1
    ;;
  xor)
    cd backend_cxx && make run
    ;;
  circle)
    cd backend_cxx && make circle
    ;;
  bench)
    cd backend_cxx && make bench
    ;;
  build)
    cd backend_cxx && make
    ;;
  *)
    echo "Usage: ./run.sh {setup|test|test-py|test-all|demo|xor|circle|bench|build|build-ext}"
    exit 1
    ;;
esac
