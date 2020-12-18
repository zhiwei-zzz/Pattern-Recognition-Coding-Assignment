'''
@File  :自动化1802+郑志伟+L5编程.py
@Author:Zhiwei Zheng
@Date  :2020/11/2 18:39
@Desc  :
'''
import numpy as np
import random
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import warnings


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def dataset_generator(x1_mean, x2_mean, y1_mean, y2_mean):
    '''

    Args:
        x1_mean:
        x2_mean:
        y1_mean:
        y2_mean:

    Returns:
        Returns:
        x1, x2: 2_dimensions normal distribution data with specific covariance

    '''
    covariance = np.array([[1, 0], [0, 1]])
    x1 = normal_2d(200, x1_mean, x2_mean, covariance)
    x2 = normal_2d(200, y1_mean, y2_mean, covariance)
    return x1, x2


def normal_2d(sample_num, x0_mean, x1_mean, Covariance):
    '''

    Args:
        sample_num:
        x0_mean:
        x1_mean:
        Covariance:

    Returns:
        dataset: 2d points

    '''

    mu = np.array([[x0_mean, x1_mean]])
    R = cholesky(Covariance)
    dataset = np.dot(np.random.randn(sample_num, 2), R) + mu
    return dataset


def logistic():
    '''

    Returns:

    '''
    w_ini = [0, 0, 0]
    lr = 0.001

    x_neg, x_pos = dataset_generator(-5, 0, 5, 0)
    random_order = random.sample(range(200), 150)
    w = w_ini

    # reconstruct dataset
    x_neg_train = []
    x_pos_train = []
    x_neg_test = []
    x_pos_test = []
    for i in range(200):
        if i in random_order:
            x_neg_train.append(x_neg[i])
            x_pos_train.append(x_pos[i])

        else:
            x_neg_test.append(x_neg[i])
            x_pos_test.append(x_pos[i])
    x_train = np.concatenate((x_neg_train, x_pos_train), axis=0)
    one_vector = np.ones((1, x_train.shape[0]))
    x_train = np.c_[one_vector.T, x_train]
    x_train_label = np.concatenate((np.ones((150, 1)) * (-1), np.ones((150, 1))), axis=0)
    x_test = np.concatenate((x_neg_test, x_pos_test), axis=0)
    one_vector = np.ones((1, x_test.shape[0]))
    x_test = np.c_[one_vector.T, x_test]
    x_test_label = np.concatenate((np.ones((50, 1)) * (-1), np.ones((50, 1))), axis=0)
    for i in range(35):
        random_order = random.sample(range(300), 300)
        for j in random_order:
            w = w - lr * sigmoid(-1 * x_train_label[j] * np.dot(w, x_train[j])) * (-1 * x_train_label[j] * x_train[j])

    plt.figure(figsize=(12, 15))
    ax = plt.subplot(111)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # y = w0 + w1x1 + w2x2
    y_line = np.arange(min(x_test[:, 1]) - 0.5, max(x_test[:, 1]) + 0.5, 0.2)
    x_line = -(w[0] / w[1]) - (w[2] * y_line / w[1])

    plt.plot(x_line, y_line, color='r', label="result")

    plt.scatter(np.array(x_test)[:, 1], np.array(x_test)[:, 2])
    error = 0
    for i in range(len(x_test)):
        result = sigmoid(np.dot(w, x_test[i].T))
        temp = result
        if result < 0.5:
            temp = 1 - temp
        plt.annotate('[' + '%.4f' % temp + ']', xy=(x_test[i][1], x_test[i][2]),
                     xytext=(x_test[i][1] + 0.1, x_test[i][2] + 0.1))

        print(x_test[i][1])
        if result > 0.5 and x_test_label[i] != 1:
            error += 1
        if result <= 0.5 and x_test_label[i] != -1:
            error += 1
    print("error: {:.2%}".format(error / 100))

    plt.savefig('./L5_' + str(1) + '.png')
    plt.show()


if __name__ == '__main__':
    logistic()
