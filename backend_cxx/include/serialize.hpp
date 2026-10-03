#pragma once
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "network.hpp"

std::string activationName(const Activation& activation);
Activation activationFromName(const std::string& name);

void saveNetwork(const NeuralNetwork& net, const std::string& filename);
NeuralNetwork loadNetwork(const std::string& filename);