import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from time import time
import datetime
from sklearn.metrics import mean_squared_error


class TensorFlow():
    def __init__(self, train_data, test_data, learning_rate, n_epochs, units):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.units = units
        self.model = 0

    def train(self):
        # Training Data
        xs_train, ys_train = self.train_data

        # determine the number of classes
        n_classes = len(np.unique(ys_train))

        # determine the shape of the input images
        in_shape = xs_train.shape[1:]

        self.image = xs_train[0]

        # define model
        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
        self.model.add(keras.layers.MaxPool2D((2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(n_classes, activation='softmax'))

        # Define Optimizer
        opt = tf.keras.optimizers.Adam(lr=self.learning_rate)

        # define loss and optimizer
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Modeling
        start_training = time()
        self.history = self.model.fit(xs_train, ys_train, epochs=self.n_epochs, batch_size=128, verbose=1)
        end_training = time()

        # Time
        duration_training = end_training - start_training

        # Number of Parameter
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        n_params = trainableParams + nonTrainableParams

        # Prediction for Training mse
        loss, error = self.model.evaluate(xs_train, ys_train, verbose=0)

        # Summary
        print('------ TensorFlow ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)
        print("Number of Parameter: ", n_params)

        return duration_training, error

    def test(self):
        # Test Data
        xs_test, ys_test = self.test_data

        xs_test = xs_test.reshape((xs_test.shape[0], xs_test.shape[1], xs_test.shape[2], 1))
        xs_test = xs_test.astype('float32') / 255.0

        # Predict Data
        start_test = time()
        loss, error = self.model.evaluate(xs_test, ys_test, verbose=0)
        end_test = time()

        # Time
        duration_test = end_test - start_test

        print(f'Duration Inference: {duration_test} seconds')

        print("Accuracy Testing: %.2f" % error)
        print("")

        return duration_test, error

    def plot(self):
        # Plot loss and val_loss
        px = 1 / plt.rcParams['figure.dpi']
        __fig = plt.figure(figsize=(800 * px, 600 * px))
        plt.plot(self.history['loss'], 'blue')
        plt.plot(self.history['val_loss'], 'red')
        plt.title('Neural Network Training loss history')
        plt.ylabel('loss (log scale)')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        plt.savefig('plots/TensorFlow_Loss-Epochs-Plot.png')
        # plt.show()
        print("TensorFlow loss Plot saved...")
        print("")
