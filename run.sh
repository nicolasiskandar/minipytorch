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

case "$1" in
  test)
    cd backend_cxx && make test
    ;;
  test-py)
    .venv/bin/python -c "$py_tests"
    ;;
  test-all)
    (cd backend_cxx && make test) || exit 1
    echo "---"
    .venv/bin/python -c "$py_tests"
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
    echo "Usage: ./run.sh {test|test-py|test-all|xor|circle|bench|build}"
    exit 1
    ;;
esac