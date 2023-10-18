# Neural Network Implementation with Python and NumPy

This repository implements a neural network for the classification of handwritten digits. The neural network is designed to work with the MNIST dataset and is capable of recognizing and classifying digits from 0 to 9 with high accuracy. It is designed to be flexible and easy to use, with a focus on understanding and implementing the fundamental concepts of neural networks

The neural network is implemented in the `neural_network.py` file. It includes backpropagation, feedforward, cost calculation, and training functions.
## Project Structure

- `main.py`: The main script that trains and tests the neural network models with different configurations and selects the best model based on test accuracy.

- `neural_network.py`: This file contains the implementation of the Neural Network model.

- `data.py`: Contains functions to load and preprocess the MNIST dataset for training and testing.

- `utils.py`: Utility functions for activation functions (ReLU, sigmoid, tanh), their derivatives, and parameter initialization.

## Dependencies

To run this project, you will need to have Python and the following libraries installed:

- NumPy
- Matplotlib
- Keras (for loading the MNIST dataset)

Install these dependencies using `pip`:

```bash
pip install numpy matplotlib keras
```

- project was built with Python 3.11.0

## Getting Started

clone this repository to your local machine:

```bash
git clone https://github.com/yaniv-simmer/machine-learning-neural-networks-exercises.git 
```

Run the `main.py` script to train and test different neural network configurations. The script will print the best model's accuracy and display a loss plot.

```bash
python main.py
```

## Neural Network Configuration

You can customize the neural network's configuration by modifying the following parameters in `main.py`:

- `hidden_layer_dimensions_lst`: List of hidden layer dimensions.

- `activation_functions_lst`: List of activation functions for each layer.

- `derivative_functions_lst`: List of derivative functions for each layer.

- `training_iterations`: Number of training iterations.

- `learning_rate`: Learning rate.

- `regularization_coefficient_lst`: List of regularization coefficients.

## Results and Visualization

The best model's configuration and test accuracy are printed to the console. The loss during training is also visualized with a loss plot.

## Contributions

Contributions, bug reports, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
