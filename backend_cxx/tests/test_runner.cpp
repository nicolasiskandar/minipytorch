#include "test_runner.hpp"

#include "test_suites.hpp"

int main() {
    TestRunner t;
    runActivationAndKernelTests(t);
    runNeuronAndLayerTests(t);
    runLossAndNetworkTests(t);
    runFastLayerAndSerializationTests(t);
    runConvAndPoolTests(t);
    t.summary();
    return t.allPassed() ? 0 : 1;
}
