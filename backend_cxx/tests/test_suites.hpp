#pragma once

class TestRunner;

void runActivationAndKernelTests(TestRunner& t);
void runNeuronAndLayerTests(TestRunner& t);
void runLossAndNetworkTests(TestRunner& t);
void runFastLayerAndSerializationTests(TestRunner& t);
