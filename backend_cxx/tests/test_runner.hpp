#pragma once
#include <cmath>
#include <iostream>
#include <string>

class TestRunner {
   public:
    void check(bool condition, const std::string& description) {
        if (condition)
            ++passed_;
        else {
            ++failed_;
            std::cout << "  FAILED: " << description << "\n";
        }
    }

    void checkNear(
        double actual,
        double expected,
        double tolerance,
        const std::string& description
    ) {
        bool ok = std::abs(actual - expected) < tolerance;
        std::string fullDescription = description;
        if (!ok)
            fullDescription += " (expected ~" + std::to_string(expected) +
                               ", got " + std::to_string(actual) + ")";
        check(ok, fullDescription);
    }

    void summary() const {
        std::cout << "\n"
                  << passed_ << " / " << (passed_ + failed_)
                  << " tests passed\n";
    }

    bool allPassed() const { return failed_ == 0; }

   private:
    int passed_ = 0;
    int failed_ = 0;
};