#pragma once

class TestRunner;

void runActivationAndKernelTests(TestRunner& t);
void runLossTests(TestRunner& t);
void runFastLayerTests(TestRunner& t);
void runConvTests(TestRunner& t);
void runPoolTests(TestRunner& t);
