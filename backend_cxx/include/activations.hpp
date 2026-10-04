#pragma once
#include <functional>

using ActivationFn = std::function<double(double)>;

struct Activation {
    ActivationFn forward;
    ActivationFn derivativeFromOutput;
};

double sigmoidFn(double z);
double sigmoidDerivFromOutput(double y);

double tanhFn(double z);
double tanhDerivFromOutput(double y);

double reluFn(double z);
double reluDerivFromOutput(double y);

extern const Activation Sigmoid;
extern const Activation Tanh;
extern const Activation ReLU;

double relu6Fn(double z);
double relu6DerivFromOutput(double y);

extern const Activation ReLU6;
