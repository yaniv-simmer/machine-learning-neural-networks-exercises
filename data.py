from keras.datasets import mnist


def load_dataset():
    print('\n\n------------- Loading MNIST dataset -------------\n')

    (training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
    print('Training images shape:', training_images.shape)
    print('Training labels shape:', training_labels.shape)
    print('Testing images shape:', testing_images.shape)
    print('Testing labels shape:', testing_labels.shape)
    print('\n\n')

    # Pre-processing
    training_images = training_images.reshape((60000, 28 * 28))
    training_images = training_images.astype('float32') / 255

    testing_images = testing_images.reshape((10000, 28 * 28))
    testing_images = testing_images.astype('float32') / 255

    input_data = training_images[:7200, :]
    output_data = training_labels[:7200]
    output_data = output_data.reshape((output_data.shape[0], 1))

    return input_data, output_data, testing_images, testing_labels

