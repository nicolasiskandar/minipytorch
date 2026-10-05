#pragma once

double sigmoidFn(double z);
double sigmoidDerivFromOutput(double y);

double tanhFn(double z);
double tanhDerivFromOutput(double y);

double reluFn(double z);
double reluDerivFromOutput(double y);

double relu6Fn(double z);
double relu6DerivFromOutput(double y);

double leakyReluFn(double z);
double leakyReluDerivFromOutput(double y);
