from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from time import time


class RandomForest:
    def __init__(self, train_data, test_data, n_estimators):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.n_estimators = n_estimators
        self.model = None

    def train(self):
        # Training Data
        xs_train, ys_train = self.train_data

        xs_train = xs_train.reshape(len(xs_train), 28 * 28)

        # normalize pixel values
        xs_train = xs_train / 255

        # Modeling
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        start_training = time()
        self.model.fit(xs_train, ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training
        duration_training = round(duration_training, 2)

        # Prediction for Training mse
        error = self.model.score(xs_train, ys_train)
        error = round(error, 2)

        # Summary
        print('------ Random Forest ------')
        print(f'Duration Training: {duration_training} seconds')
        print('Accuracy Training: ', error)

        return duration_training, error

    def test(self):
        # Test Data
        xs_test, ys_test = self.test_data

        # normalize pixel values
        xs_test = xs_test / 255

        xs_test = xs_test.reshape(len(xs_test), 28 * 28)

        # Predict Data
        start_test = time()
        error = self.model.score(xs_test, ys_test)
        error = round(error, 2)
        end_test = time()

        # Time
        duration_test = end_test - start_test
        duration_test = round(duration_test, 2)

        print(f'Duration Inference: {duration_test} seconds')

        print("Accuracy Testing: %.2f" % error)
        print("")

        return duration_test, error

    def plot(self):
        # Plot loss and val_loss
        px = 1 / plt.rcParams['figure.dpi']
        __fig = plt.figure(figsize=(800 * px, 600 * px))
        plt.plot(self.history.history['loss'], 'blue')
        plt.plot(self.history.history['val_loss'], 'red')
        plt.title('Neural Network Training loss history')
        plt.ylabel('loss (log scale)')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        url = f"plots/training-history/TensorFlow_{self.id}_Loss-Epochs-Plot.png"
        plt.savefig(url)
        # plt.show()
        print("TensorFlow loss Plot saved...")
        print("")
