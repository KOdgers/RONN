from tensorflow import keras

def load_test_model(inshape, outshape):
    input = keras.Input(shape=(inshape))
    X = keras.layers.Dense(5, activation='relu')(input)
    X = keras.layers.Dense(5, activation='relu')(X)
    # X = keras.layers.Dense(5, activation='relu')(X)
    # X = keras.layers.Dense(10, activation='relu')(X)
    out = keras.layers.Dense(outshape, activation='softmax')(X)

    return input, out


def load_test_cnn_mnist(inshape, outshape):
    input = keras.Input(shape=(inshape))
    corrected = keras.layers.Reshape((28, 28, 1))(input)
    X = keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(corrected)
    X = keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    X = keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dropout(.5)(X)
    out = keras.layers.Dense(outshape, activation='softmax')(X)

    return input, out