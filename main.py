import data
import neural_network
import utils

X, y, test_images, test_labels = data.load_data()

layers = [32, 64]
activations = [utils.sigmoid, utils.ReLU, utils.sigmoid]
epochs = 100
alpha = 0.2
Lambda = 0.0
NN = neural_network.model(layers, X, y, activations, epochs, alpha, Lambda)
NN.train()
