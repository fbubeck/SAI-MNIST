from random import random
from algorithms import TensorFlow
from algorithms import PolyRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import SampleData
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
    sampleData = SampleData.SampleData(n_numbers, min_bias, max_bias)
    train_data = sampleData.get_Data()
    test_data = sampleData.get_Data()

    # Data Exploration
    exploration = Exploration.Exploration(train_data, test_data)
    exploration.plot()

    # Creating Algorithm Objects
    tensorFlow = TensorFlow.TensorFlow(train_data, test_data, tf_learning_rate, tf_n_epochs, tf_units)
    tensorFlow2 = TensorFlow.TensorFlow(train_data, test_data, tf2_learning_rate, tf2_n_epochs, tf2_units)
    tensorFlow3 = TensorFlow.TensorFlow(train_data, test_data, tf3_learning_rate, tf3_n_epochs, tf3_units)
    
    linearRegression = PolyRegression.PolyRegression(train_data, test_data, lr_degree)
    linearRegression2 = PolyRegression.PolyRegression(train_data, test_data, lr2_degree)
    linearRegression3 = PolyRegression.PolyRegression(train_data, test_data, lr3_degree)
    
    decisionTree = DecisionTree.DecisionTree(train_data, test_data, dt_max_depth)
    decisionTree2 = DecisionTree.DecisionTree(train_data, test_data, dt2_max_depth)
    decisionTree3 = DecisionTree.DecisionTree(train_data, test_data, dt3_max_depth)
    
    randomForest = RandomForestRegressor.RandomForest(train_data, test_data, rf_n_estimators, rf_random_state)
    randomForest2 = RandomForestRegressor.RandomForest(train_data, test_data, rf2_n_estimators, rf2_random_state)
    randomForest3 = RandomForestRegressor.RandomForest(train_data, test_data, rf3_n_estimators, rf3_random_state)

    # Tensorflow
    tensorFlow_training_duration, tensorFlow_training_error  = tensorFlow.train()
    tensorFlow_test_duration, tensorFlow_test_error, tensorFlow_y_pred  = tensorFlow.test()
    tensorFlow.plot()
    tensorFlow2_training_duration, tensorFlow2_training_error  = tensorFlow2.train()
    tensorFlow2_test_duration, tensorFlow2_test_error, tensorFlow2_y_pred  = tensorFlow2.test()
    tensorFlow2.plot()
    tensorFlow3_training_duration, tensorFlow3_training_error  = tensorFlow3.train()
    tensorFlow3_test_duration, tensorFlow3_test_error, tensorFlow3_y_pred  = tensorFlow3.test()
    tensorFlow3.plot()

    # Linear Regression
    linearRegression_training_duration, linearRegression_training_error  = linearRegression.train()
    linearRegression_test_duration, linearRegression_test_error, linearRegression_y_pred = linearRegression.test()
    linearRegression2_training_duration, linearRegression2_training_error  = linearRegression2.train()
    linearRegression2_test_duration, linearRegression2_test_error, linearRegression2_y_pred = linearRegression2.test()
    linearRegression3_training_duration, linearRegression3_training_error  = linearRegression3.train()
    linearRegression3_test_duration, linearRegression3_test_error, linearRegression3_y_pred = linearRegression3.test()

    # Decision Tree
    decisionTree_training_duration, decisionTree_training_error  = decisionTree.train()
    decisionTree_test_duration, decisionTree_test_error, decisionTree_y_pred = decisionTree.test()
    decisionTree2_training_duration, decisionTree2_training_error  = decisionTree2.train()
    decisionTree2_test_duration, decisionTree2_test_error, decisionTree2_y_pred = decisionTree2.test()
    decisionTree3_training_duration, decisionTree3_training_error  = decisionTree3.train()
    decisionTree3_test_duration, decisionTree3_test_error, decisionTree3_y_pred = decisionTree3.test()

    # Random Forest
    randomForest_training_duration, randomForest_training_error = randomForest.train()
    randomForest_test_duration, randomForest_test_error, randomForest_y_pred  = randomForest.test()
    randomForest2_training_duration, randomForest2_training_error = randomForest2.train()
    randomForest2_test_duration, randomForest2_test_error, randomForest2_y_pred  = randomForest2.test()
    randomForest3_training_duration, randomForest3_training_error = randomForest3.train()
    randomForest3_test_duration, randomForest3_test_error, randomForest3_y_pred  = randomForest3.test()

    # Plots
    xs_test = test_data[0]
    ys_test = test_data[1]

    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1200*px, 800*px))
    fig.suptitle('Model Comparison')
    axs1 = fig.add_subplot(221)
    axs2 = fig.add_subplot(222)
    axs3 = fig.add_subplot(223)
    axs4 = fig.add_subplot(224)
    axs1.scatter(xs_test, ys_test, color='b', s=1, label="Test Data", alpha=0.5)
    axs1.scatter(xs_test, tensorFlow_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
    axs2.scatter(xs_test, ys_test, color='b', s=1, label="Test Data", alpha=0.5)
    axs2.scatter(xs_test, linearRegression_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
    axs3.scatter(xs_test, ys_test, color='b', s=1, label="Test Data", alpha=0.5)
    axs3.scatter(xs_test, decisionTree_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
    axs4.scatter(xs_test, ys_test, color='b', s=1, label="Test Data", alpha=0.5)
    axs4.scatter(xs_test, randomForest_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
    axs1.title.set_text('TensorFlow Model')
    axs2.title.set_text('Linear Regression Model')
    axs3.title.set_text('Decision Tree Model')
    axs4.title.set_text('Random Forest Model')
    axs1.legend(loc="upper left", markerscale=10, scatterpoints=1)
    axs2.legend(loc="upper left", markerscale=10, scatterpoints=1)
    axs3.legend(loc="upper left", markerscale=10, scatterpoints=1)
    axs4.legend(loc="upper left", markerscale=10, scatterpoints=1)
    axs1.text(.95, 0.05, ("Error: " + str(tensorFlow_test_error)), ha='right', va='bottom', transform=axs1.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
    axs2.text(.95, 0.05, ("Error: " + str(linearRegression_test_error)), ha='right', va='bottom', transform=axs2.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
    axs3.text(.95, 0.05, ("Error: " + str(decisionTree_test_error)), ha='right', va='bottom', transform=axs3.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
    axs4.text(.95, 0.05, ("Error: " + str(randomForest_test_error)), ha='right', va='bottom', transform=axs4.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
    plt.savefig('plots/Algorithms_Model_Comparison.png')
    #plt.show()

    tf_training_x = [tensorFlow_training_error, tensorFlow2_training_error, tensorFlow3_training_error]
    lr_training_x = [linearRegression_training_error, linearRegression2_training_error, linearRegression3_training_error]
    dt_training_x = [decisionTree_training_error, decisionTree2_training_error, decisionTree3_training_error]
    rf_training_x = [randomForest_training_error, randomForest2_training_error, randomForest3_training_error]
    tf_training_y = [tensorFlow_training_duration, tensorFlow2_training_duration, tensorFlow3_training_duration]
    lr_training_y = [linearRegression_training_duration, linearRegression2_training_duration, linearRegression3_training_duration]
    dt_training_y = [decisionTree_training_duration, decisionTree2_training_duration, decisionTree3_training_duration]
    rf_training_y = [randomForest_training_duration, randomForest2_training_duration, randomForest3_training_duration]

    tf_test_x = [tensorFlow_test_error, tensorFlow2_test_error, tensorFlow3_test_error]
    lr_test_x = [linearRegression_test_error, linearRegression2_test_error, linearRegression3_test_error]
    dt_test_x = [decisionTree_test_error, decisionTree2_test_error, decisionTree3_test_error]
    rf_test_x = [randomForest_test_error, randomForest2_test_error,randomForest3_test_error]
    tf_test_y = [tensorFlow_test_duration, tensorFlow2_test_duration,tensorFlow3_test_duration]
    lr_test_y = [linearRegression_test_duration, linearRegression2_test_duration, linearRegression3_test_duration]
    dt_test_y = [decisionTree_test_duration, decisionTree2_test_duration, decisionTree3_test_duration]
    rf_test_y = [randomForest_test_duration, randomForest2_test_duration, randomForest3_test_duration]

    fig = plt.figure(figsize=(1200*px, 800*px))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_ylabel('Duration [in seconds; log scale]')
    ax1.set_xlabel('Model-Performance (Error)')
    ax2.set_ylabel('Duration [in seconds; log scale]')
    ax2.set_xlabel('Model-Performance (Error)')
    fig.suptitle('Efficiency of different ML-Algorithms and Parametersets')
    ax1.plot(tf_training_x, tf_training_y, '-o', c='blue', alpha=0.6)
    ax2.plot(tf_test_x, tf_test_y, '-o', c='blue', alpha=0.6)
    ax1.plot(lr_training_x, lr_training_y, '-o', c='red', alpha=0.6)
    ax2.plot(lr_test_x, lr_test_y, '-o', c='red', alpha=0.6)
    ax1.plot(dt_training_x, dt_training_y, '-o', c='green', alpha=0.6)
    ax2.plot(dt_test_x, dt_test_y, '-o', c='green', alpha=0.6)
    ax1.plot(rf_training_x, rf_training_y, '-o', c='orange', alpha=0.6)
    ax2.plot(rf_test_x, rf_test_y, '-o', c='orange', alpha=0.6)
    ax1.title.set_text('Training')
    ax2.title.set_text('Inference')
    plt.legend(["TensorFlow Neural Network", "Linear/Poly Regression", "Decision Tree Regressor", "Random Forest Regressor"], loc='lower center', ncol=4, bbox_transform=fig.transFigure, bbox_to_anchor=(0.5,0))
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.savefig('plots/Algorithms_Evaluation.png')
    #plt.show()
    print("Evaluation Plot saved...")
    print("")



if __name__ == "__main__":
    main()
