import tensorflow as tf


class DataProvider():

    def get_Data(self):
        print("Preprocess data ...")
        mnist = tf.keras.datasets.mnist

        (xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()
        xs_train, xs_test = xs_train / 255.0, xs_test / 255.0

        # reshape data to have a single channel
        xs_train = xs_train.reshape((xs_train.shape[0], xs_train.shape[1], xs_train.shape[2], 1))

        # normalize pixel values
        xs_train = xs_train.astype('float32') / 255.0

        train_data = xs_train, ys_train
        test_data = xs_test, ys_test

        return train_data, test_data
