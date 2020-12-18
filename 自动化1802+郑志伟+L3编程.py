'''
@File  :自动化1802+郑志伟+L3编程.py
@Author:Zhiwei Zheng
@Date  :2020/10/31 10:17
@Desc  :
'''

import numpy as np
import random
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import warnings


def SSErr(input_data, input_gt):
    '''
    Args:
        input_data:[N * d]
        input_gt:[N * 1]

    Returns:
        w: weight vector [d * 1]

    Describe:
        采用广义逆

    '''

    x_num = input_data.shape[0]
    ones_vector = np.ones((1, x_num))
    input_data = np.c_[ones_vector.T, input_data]
    w = np.dot(np.linalg.pinv(input_data), input_gt.reshape((len(input_gt), 1)))
    return w


def LMSalg(input_data, input_gt, w_ini, lr):
    '''

    Args:
        input_data:[N * d]
        input_gt:[N * 1]
        w_ini: [d * 1]

    Returns:
        w: weight vector [d * 1]
    '''

    x_num = input_data.shape[0]
    ones_vector = np.ones((1, x_num))
    input_data = np.c_[ones_vector.T, input_data]
    w = w_ini
    for i in range(25):
        random_order = random.sample(range(x_num), x_num)  # randomly choosing data
        for j in random_order:  # exploit all dataset
            a = np.dot(w.T, input_data[j])
            delta = (a - input_gt[j]) * input_data[j].reshape((len(input_data[j]), 1))
            if delta.all() == 0:
                return w
            w = (w - lr * delta)
    return w


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


def evaluate_w(w_weight, input_data, input_gt, precision):
    label_0 = label_1 = 0
    x_num = input_data.shape[0]
    ones_vector = np.ones((1, x_num))
    input_data = np.c_[ones_vector.T, input_data]
    for j in range(len(input_data)):
        compare = np.sign(np.dot(w_weight.T, input_data[j])) != input_gt[j]
        if input_gt[j] == -1 and not compare:
            label_0 += 1
        if input_gt[j] == 1 and not compare:
            label_1 += 1

    precision[0] = label_0 / len(input_data) * 2
    precision[1] = label_1 / len(input_data) * 2
    precision[2] = (label_0 + label_1) / len(input_data)
    return precision


def test(x1_mean, x2_mean, y1_mean, y2_mean, w_ini, lr, figure_num):
    x_neg, x_pos = dataset_generator(x1_mean, x2_mean, y1_mean, y2_mean)
    x_neg_gt = np.ones(len(x_neg)) * -1
    x_pos_gt = np.ones(len(x_pos))
    x_gt = np.concatenate((x_neg_gt, x_pos_gt), axis=0)
    x = np.concatenate((x_neg, x_pos), axis=0)
    w_sser = SSErr(x, x_gt)
    w_lmsalg = LMSalg(x, x_gt, w_ini, lr)
    w_sser_pre = evaluate_w(w_sser, x, x_gt, [0, 0, 0])
    w_lmsalg_pre = evaluate_w(w_lmsalg, x, x_gt, [0, 0, 0])
    print("The sser weights of the model trained on Problem%d dataset is " % figure_num, w_sser, )
    print("The sser model precision of label'-1', label'1', and the whole dataset is {:.2%} {:.2%} {:.2%}.".format(
        w_sser_pre[0],
        w_sser_pre[1],
        w_sser_pre[2]))
    print("The lmsalg weights of the model trained on Problem%d dataset is " % figure_num, w_lmsalg, )
    print("The lmsalg model precision of label'-1', label'1', and the whole dataset is {:.2%} {:.2%} {:.2%}.".format(
        w_lmsalg_pre[0],
        w_lmsalg_pre[1],
        w_lmsalg_pre[2]))

    plt.figure(figure_num, figsize=(16, 12))

    ax = plt.subplot(1, 2, 1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    y_line = np.arange(min(x[:, 1]) - 0.5, max(x[:, 1]) + 0.5, 0.2)
    x_line = -(w_sser[0] / w_sser[1]) - (w_sser[2] * y_line / w_sser[1])
    plt.plot(x_neg[:, 0], x_neg[:, 1], '+')
    plt.plot(x_pos[:, 0], x_pos[:, 1], '+')
    plt.plot(x_line, y_line, color='r', label="w_sser_result")

    ax = plt.subplot(1, 2, 2)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    y_line = np.arange(min(x[:, 1]) - 0.5, max(x[:, 1]) + 0.5, 0.2)
    x_line = -(w_lmsalg[0] / w_lmsalg[1]) - (w_lmsalg[2] * y_line / w_lmsalg[1])
    plt.plot(x_neg[:, 0], x_neg[:, 1], '+')
    plt.plot(x_pos[:, 0], x_pos[:, 1], '+')
    plt.plot(x_line, y_line, color='r', label="w_lmsalg_result")
    plt.savefig('./L3_' + str(figure_num) + '.png')
    plt.show()


if __name__ == '__main__':
    w_ini = np.ones((3, 1))
    test(-5, 0, 5, 0, w_ini, 0.001, 3)
    test(-2, 0, 2, 0, w_ini, 0.001, 4)
    test(-1, 0, 1, 0, w_ini, 0.001, 5)

# output
# The sser weights of the model trained on Problem3 dataset is  [[-0.00052543]
#  [ 0.19044879]
#  [ 0.0121187 ]]
# The sser model precision of label'-1', label'1', and the whole dataset is 100.00% 100.00% 100.00%.
# The lmsalg weights of the model trained on Problem3 dataset is  [[-0.00302901]
#  [ 0.18658061]
#  [ 0.0127614 ]]
# The lmsalg model precision of label'-1', label'1', and the whole dataset is 100.00% 100.00% 100.00%.
# The sser weights of the model trained on Problem4 dataset is  [[-0.04458755]
#  [ 0.39462652]
#  [ 0.02410489]]
# The sser model precision of label'-1', label'1', and the whole dataset is 97.50% 96.00% 96.75%.
# The lmsalg weights of the model trained on Problem4 dataset is  [[-0.04334527]
#  [ 0.40031952]
#  [ 0.02372101]]
# The lmsalg model precision of label'-1', label'1', and the whole dataset is 97.50% 96.00% 96.75%.
# The sser weights of the model trained on Problem5 dataset is  [[-0.02862249]
#  [ 0.50030749]
#  [ 0.0150165 ]]
# The sser model precision of label'-1', label'1', and the whole dataset is 84.00% 83.50% 83.75%.
# The lmsalg weights of the model trained on Problem5 dataset is  [[-0.02650639]
#  [ 0.49969523]
#  [ 0.01728268]]
# The lmsalg model precision of label'-1', label'1', and the whole dataset is 84.00% 83.50% 83.75%.
