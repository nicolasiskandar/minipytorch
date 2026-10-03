#!/bin/bash
case "$1" in
  test)
    cd /home/nicolas/workspace/minipytorch/backend_cxx && make test
    ;;
  test-py)
    PYTHONPATH=/home/nicolas/workspace/minipytorch:/home/nicolas/workspace/minipytorch/.venv/lib/python3.11/site-packages \
    /home/nicolas/.local/bin/python3.11 -c "
import sys, os, glob
sys.path.insert(0, '/home/nicolas/workspace/minipytorch')
for f in sorted(glob.glob('/home/nicolas/workspace/minipytorch/tests_py/*.py')):
    if f.endswith('__init__.py'): continue
    mod = os.path.basename(f)[:-3]
    m = __import__('tests_py.'+mod, fromlist=[''])
    for k in dir(m):
        if k.startswith('test_'): getattr(m,k)()
print('all python tests passed')
"
    ;;
  test-all)
    cd /home/nicolas/workspace/minipytorch/backend_cxx && make test
    echo "---"
    PYTHONPATH=/home/nicolas/workspace/minipytorch:/home/nicolas/workspace/minipytorch/.venv/lib/python3.11/site-packages \
    /home/nicolas/.local/bin/python3.11 -c "
import sys, os, glob
sys.path.insert(0, '/home/nicolas/workspace/minipytorch')
for f in sorted(glob.glob('/home/nicolas/workspace/minipytorch/tests_py/*.py')):
    if f.endswith('__init__.py'): continue
    mod = os.path.basename(f)[:-3]
    m = __import__('tests_py.'+mod, fromlist=[''])
    for k in dir(m):
        if k.startswith('test_'): getattr(m,k)()
print('all python tests passed')
"
    ;;
  xor)
    cd /home/nicolas/workspace/minipytorch/backend_cxx && make run
    ;;
  circle)
    cd /home/nicolas/workspace/minipytorch/backend_cxx && make circle
    ;;
  bench)
    cd /home/nicolas/workspace/minipytorch/backend_cxx && make bench
    ;;
  build)
    cd /home/nicolas/workspace/minipytorch/backend_cxx && make
    ;;
  *)
    echo "Usage: ./run.sh {test|test-py|test-all|xor|circle|bench|build}"
    exit 1
    ;;
esac
