import data
import neural_network
import utils

input_data, target_labels, test_images, test_labels = data.load_dataset()

hidden_layer_dimensions = [32]
activation_functions = [utils.ReLU, utils.sigmoid]
derivative_functions = [utils.dReLU, utils.d_sigmoid]
training_iterations = 60
learning_rate = 0.5
regularization_coefficient = 1
NN = neural_network.model(hidden_layer_dimensions, input_data, target_labels, activation_functions,
                          derivative_functions, training_iterations, learning_rate, regularization_coefficient)
NN.train_neural_network()
# score = NN.evaluate(test_images, test_labels)
