from tensorflow import keras

class simple_seq_model:

    def __init__(self):
        self.name='Simple_Seq'
        self.version = '0.1'
        self.valid_data = ['load_data_covtype','load_data_kddcup']

    def build(self,inshape,outshape):

        inputl = keras.Input(shape=(inshape))
        X = keras.layers.Dense(10, activation='relu')(inputl)
        X = keras.layers.Dense(10, activation='relu')(X)
        X = keras.layers.Dense(10, activation='relu')(X)
        out = keras.layers.Dense(outshape, activation='softmax')(X)

        return inputl, out


class simple_seq_cnn_flat:

    def __init__(self):
        self.name = 'Simple_Seq_Cnn_Flat'
        self.version = '0.1'
        self.valid_data = ['load_data_mnist']


    def build(self,inshape, outshape):
        inputl = keras.Input(shape=(inshape))
        corrected = keras.layers.Reshape((28, 28, 1))(inputl)
        X = keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(corrected)
        X = keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
        X = keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(X)
        X = keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dropout(.5)(X)
        out = keras.layers.Dense(outshape, activation='softmax')(X)
        return inputl, out




