#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

extern "C" double
dotProductAsm(const double* a, const double* b, std::size_t n);

double dotProductCpp(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

int main() {
    std::vector<double> a = {2.0, -1.0, 0.5};
    std::vector<double> b = {3.0, 4.0, 2.0};
    std::size_t n = a.size();

    double resultAsm = dotProductAsm(a.data(), b.data(), n);
    double resultCpp = dotProductCpp(a.data(), b.data(), n);

    std::cout << "C++ result:  " << resultCpp << "\n";
    std::cout << "asm result:  " << resultAsm << "\n";
    std::cout << "difference:  " << std::abs(resultAsm - resultCpp) << "\n";

    const int iters = 500;
    double sink = 0.0;

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
        sink += dotProductCpp(a.data(), b.data(), n);
    auto t1 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
        sink += dotProductAsm(a.data(), b.data(), n);
    auto t2 = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> cppMs = t1 - t0;
    std::chrono::duration<double, std::milli> asmMs = t2 - t1;

    std::cout << "g++ -O2 compiled dotProduct: " << cppMs.count() << " ms ("
              << cppMs.count() / iters << " ms/iter)\n";

    std::cout << "hand-written assembly:       " << asmMs.count() << " ms ("
              << asmMs.count() / iters << " ms/iter)\n";

    std::cout << "ratio (asm time / cpp time): "
              << asmMs.count() / cppMs.count() << "\n";

    std::cout << "(sink=" << sink << ")\n";

    return 0;
}