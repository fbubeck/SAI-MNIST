from random import random
from algorithms import TensorFlow
from algorithms import PolyRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import DataProvider
from data import Exploration
import json
from matplotlib import pyplot as plt


def main():
    print("Starting...")
    print("")

    # read config.json
    with open('config/config.json') as file:
        config = json.load(file)

    # Get Parameters from config file
    n_numbers = config["GlobalParameters"]["n_numbers"]
    min_bias = config["GlobalParameters"]["min_bias"]
    max_bias = config["GlobalParameters"]["max_bias"]

    tf_learning_rate = config["TensorFlow"]["learning_rate"]
    tf_n_epochs = config["TensorFlow"]["n_epochs"]
    tf_units = config["TensorFlow"]["n_units"]
    tf2_learning_rate = config["TensorFlow2"]["learning_rate"]
    tf2_n_epochs = config["TensorFlow2"]["n_epochs"]
    tf2_units = config["TensorFlow2"]["n_units"]
    tf3_learning_rate = config["TensorFlow3"]["learning_rate"]
    tf3_n_epochs = config["TensorFlow3"]["n_epochs"]
    tf3_units = config["TensorFlow3"]["n_units"]
    
    lr_degree = config["PolyRegression"]["degree"]
    lr2_degree = config["PolyRegression2"]["degree"]
    lr3_degree = config["PolyRegression3"]["degree"]

    dt_max_depth = config["DecisionTree"]["max_depth"]
    dt2_max_depth = config["DecisionTree2"]["max_depth"]
    dt3_max_depth = config["DecisionTree3"]["max_depth"]

    rf_n_estimators = config["RandomForest"]["n_estimators"]
    rf_random_state = config["RandomForest"]["random_state"]
    rf2_n_estimators = config["RandomForest2"]["n_estimators"]
    rf2_random_state = config["RandomForest2"]["random_state"]
    rf3_n_estimators = config["RandomForest3"]["n_estimators"]
    rf3_random_state = config["RandomForest3"]["random_state"]

    # Get Sample Data
    sampleData = DataProvider.DataProvider()
    train_data, test_data = sampleData.get_Data()

    # Creating Algorithm Objects
    algo1_a = TensorFlow.TensorFlow(train_data, test_data, tf_learning_rate, tf_n_epochs, tf_units)
    algo1_b = TensorFlow.TensorFlow(train_data, test_data, tf2_learning_rate, tf2_n_epochs, tf2_units)
    algo1_c = TensorFlow.TensorFlow(train_data, test_data, tf3_learning_rate, tf3_n_epochs, tf3_units)

    # Tensorflow
    algo1_a_trainingDuration, algo1_a_trainingError = algo1_a.train()
    algo1_a_testDuration, algo1_a_testError = algo1_a.test()
    #algo1_a.plot()
    algo1_b_trainingDuration, algo1_b_trainingError = algo1_b.train()
    algo1_b_testDuration, algo1_b_testError = algo1_b.test()
    algo1_c_trainingDuration, algo1_c_trainingError = algo1_c.train()
    algo1_c_testDuration, algo1_c_testError = algo1_c.test()

    # Plots
    xs_test = test_data[0]
    ys_test = test_data[1]

    px = 1/plt.rcParams['figure.dpi']

    algo1_training_x = [algo1_a_trainingError, algo1_b_trainingError, algo1_c_trainingError]
    algo1_training_y = [algo1_a_trainingDuration, algo1_b_trainingDuration, algo1_c_trainingDuration]
    algo1_test_x = [algo1_a_testError, algo1_b_testError, algo1_c_testError]
    algo1_test_y = [algo1_a_testDuration, algo1_b_testDuration, algo1_c_testDuration,]


    fig = plt.figure(figsize=(1200*px, 800*px))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_ylabel('Duration [in seconds; log scale]')
    ax1.set_xlabel('Model-Performance (Error)')
    ax2.set_ylabel('Duration [in seconds; log scale]')
    ax2.set_xlabel('Model-Performance (Error)')
    fig.suptitle('Efficiency of different ML-Algorithms and Parametersets')
    ax1.plot(algo1_training_x, algo1_training_y, '-o', c='blue', alpha=0.6)
    ax2.plot(algo1_test_x, algo1_test_y, '-o', c='blue', alpha=0.6)
    ax1.title.set_text('Training')
    ax2.title.set_text('Inference')
    plt.legend(["TensorFlow Neural Network (3 different Parametersets)"], loc='lower center', ncol=4, bbox_transform=fig.transFigure, bbox_to_anchor=(0.5,0))
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.savefig('plots/Algorithms_Evaluation.png')
    #plt.show()
    print("Evaluation Plot saved...")
    print("")



if __name__ == "__main__":
    main()
