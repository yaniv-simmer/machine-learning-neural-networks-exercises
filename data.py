from keras.datasets import mnist


def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    print('train_images.shape =', train_images.shape)
    print('train_labels.shape =', train_labels.shape)
    print('test_images.shape =', test_images.shape)
    print('test_labels.shape =', test_labels.shape)

    # pre-processing
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    X = train_images[:7200, :]
    y = train_labels[:7200]
    y = y.reshape((y.shape[0], 1))

    return X, y, test_images, test_labels
