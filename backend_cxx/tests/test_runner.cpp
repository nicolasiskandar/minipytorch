#include "test_runner.hpp"

#include "test_suites.hpp"

int main() {
    TestRunner t;
    runActivationAndKernelTests(t);
    runLossTests(t);
    runFastLayerTests(t);
    runConvTests(t);
    runPoolTests(t);
    t.summary();
    return t.allPassed() ? 0 : 1;
}
