CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -O2

CPP_SRC := src/activations.cpp src/fastLayer.cpp src/layer.cpp src/losses.cpp \
	src/network.cpp src/neuron.cpp src/serialize.cpp
ASM_SRC := asm/kernels/dot_product.S asm/kernels/activations.S \
	asm/kernels/fast_layer.S asm/kernels/updates.S \
	asm/kernels/neuron.S asm/kernels/losses.S
LIB_SRC := $(CPP_SRC) $(ASM_SRC)

TEST_SRC := tests/test_runner.cpp tests/*_tests.cpp $(LIB_SRC)
TEST_BIN := nn_test

XOR_SRC := experiments/main_xor.cpp $(LIB_SRC)
XOR_BIN := nn_xor

CIRCLE_SRC := experiments/main_circle.cpp $(LIB_SRC)
CIRCLE_BIN := nn_circle

BENCH_SRC := experiments/check.cpp $(LIB_SRC)
BENCH_BIN := nn_bench

all: $(TEST_BIN) $(XOR_BIN) $(CIRCLE_BIN) $(BENCH_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -Itests $(TEST_SRC) -o $(TEST_BIN)

$(XOR_BIN): $(XOR_SRC)
	$(CXX) $(CXXFLAGS) $(XOR_SRC) -o $(XOR_BIN)

$(CIRCLE_BIN): $(CIRCLE_SRC)
	$(CXX) $(CXXFLAGS) $(CIRCLE_SRC) -o $(CIRCLE_BIN)

$(BENCH_BIN): $(BENCH_SRC)
	$(CXX) $(CXXFLAGS) $(BENCH_SRC) -o $(BENCH_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

run: $(XOR_BIN)
	./$(XOR_BIN)

circle: $(CIRCLE_BIN)
	./$(CIRCLE_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -f $(TEST_BIN) $(XOR_BIN) $(CIRCLE_BIN) $(BENCH_BIN)
