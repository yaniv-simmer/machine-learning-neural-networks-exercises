from utils import ReLU, sigmoid, d_ReLU, d_sigmoid
import neural_network
import data


def main():
    """
    trains and tests the neural network models with different configurations.
    the best model is selected based on the test accuracy.
    """

    # load the MNIST dataset
    input_data, target_labels, test_images, test_labels = data.load_dataset()

    # different configurations for the neural networks
    hidden_layer_dimensions_lst = [[32], [32, 64], [16, 32]]
    activation_functions_lst = [[ReLU, sigmoid],
                                [ReLU, ReLU, sigmoid],
                                [ReLU, sigmoid, sigmoid]]
    derivative_functions_lst = [[d_ReLU, d_sigmoid],
                                [d_ReLU, d_ReLU, d_sigmoid],
                                [d_ReLU, d_sigmoid, d_sigmoid]]
    training_iterations = 100
    learning_rate = [0.1, 0.15, 0.2]
    regularization_coefficient_lst = [1, 0.1, 1]

    # train and test the models and find the best model
    best_model_config = {'model': None, 'accuracy': 0}
    for i in range(3):
        NN = neural_network.model(hidden_layer_dimensions_lst[i],
                                  input_data, target_labels,
                                  activation_functions_lst[i],
                                  derivative_functions_lst[i],
                                  training_iterations, learning_rate[i],
                                  regularization_coefficient_lst[i])
        
        NN.train_neural_network()
        _, accuracy = NN.predict_output(test_images, test_labels)
        
        print("Model test accuracy: ", accuracy, '\n\n')
        if accuracy > best_model_config['accuracy']:
            best_model_config['model'], best_model_config['accuracy'] = NN, accuracy


    # print the best model configuration and plot the loss
    best_NN_model = best_model_config['model']

    print('\n\n\n--------------------------------------\n')
    print("Best Neural Network model accuracy: ", best_model_config['accuracy'], '\n\nBest', end=' ')
    best_NN_model.print_network_configuration()
    best_NN_model.plot_loss()


if __name__ == '__main__':
    main()
