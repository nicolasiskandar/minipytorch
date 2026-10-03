#include <iostream>
#include <random>

#include "layer.hpp"
#include "losses.hpp"
#include "network.hpp"
#include "neuron.hpp"

int main() {
    std::mt19937 dataRng(7);
    std::uniform_real_distribution<double> coord(-1.5, 1.5);

    std::vector<std::vector<double>> allInputs;
    std::vector<std::vector<double>> allTargets;
    for (int i = 0; i < 300; ++i) {
        double x1 = coord(dataRng);
        double x2 = coord(dataRng);
        double label = (x1 * x1 + x2 * x2 < 1.0) ? 1.0 : 0.0;
        allInputs.push_back({x1, x2});
        allTargets.push_back({label});
    }

    std::vector<std::vector<double>> trainInputs(
        allInputs.begin(), allInputs.begin() + 240
    );
    std::vector<std::vector<double>> trainTargets(
        allTargets.begin(), allTargets.begin() + 240
    );
    std::vector<std::vector<double>> testInputs(
        allInputs.begin() + 240, allInputs.end()
    );
    std::vector<std::vector<double>> testTargets(
        allTargets.begin() + 240, allTargets.end()
    );

    std::mt19937 wRng(3);
    std::uniform_real_distribution<double> wDist(-1.0, 1.0);

    std::vector<Neuron> hiddenNeurons;
    for (int i = 0; i < 8; ++i)
        hiddenNeurons.push_back(
            Neuron({wDist(wRng), wDist(wRng)}, wDist(wRng), Tanh)
        );

    std::vector<double> outW;
    for (int i = 0; i < 8; ++i) outW.push_back(wDist(wRng));

    NeuralNetwork net(
        {Layer(hiddenNeurons), Layer({Neuron(outW, wDist(wRng), Sigmoid)})}
    );

    const double learningRate = 0.1;
    const int epochs = 3000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        for (std::size_t i = 0; i < trainInputs.size(); ++i)
            totalLoss += net.trainStep(
                trainInputs[i], trainTargets[i], learningRate,
                binaryCrossEntropy
            );

        if (epoch % 500 == 0 || epoch == epochs - 1)
            std::cout << "epoch " << epoch
                      << "  avg train loss = " << totalLoss / trainInputs.size()
                      << "\n";
    }

    int correct = 0;
    for (std::size_t i = 0; i < testInputs.size(); ++i) {
        double pred = net.predict(testInputs[i])[0];
        double target = testTargets[i][0];
        if ((pred >= 0.5) == (target == 1.0)) ++correct;
    }

    std::cout << "\nTest accuracy: " << correct << "/" << testInputs.size()
              << " (" << 100.0 * correct / testInputs.size() << "%)\n";

    return 0;
}