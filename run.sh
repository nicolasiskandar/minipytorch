#!/bin/bash
cd "$(dirname "$0")" || exit 1

py_tests='
import sys, os, glob
sys.path.insert(0, os.getcwd())
for f in sorted(glob.glob("tests_py/*.py")):
    if f.endswith("__init__.py"): continue
    mod = os.path.basename(f)[:-3]
    m = __import__("tests_py."+mod, fromlist=[""])
    for k in dir(m):
        if k.startswith("test_"): getattr(m,k)()
print("all python tests passed")
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