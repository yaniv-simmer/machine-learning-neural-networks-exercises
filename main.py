import data
import neural_network
import utils

X, y, test_images, test_labels = data.load_dataset()

layers = [32]
activations = [utils.ReLU, utils.sigmoid]
dirvatives = [utils.dReLU, utils.d_sigmoid]
epochs = 60
alpha = 0.5
Lambda = 1
NN = neural_network.model(layers, X, y, activations,
                          dirvatives, epochs, alpha, Lambda)
NN.train()
# score = NN.evaluate(test_images, test_labels)
