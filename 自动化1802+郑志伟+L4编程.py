'''
@File  :自动化1802+郑志伟+L4编程.py
@Author:Zhiwei Zheng
@Date  :2020/10/31 9:25
@Desc  :
'''
import numpy as np
import random
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import warnings
import seaborn as sns


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


def Fisher():
    x_pos, x_neg = dataset_generator(-5, 0, 5, 0)
    x_pos_mean = np.mean(x_pos, axis=0)
    x_neg_mean = np.mean(x_neg, axis=0)
    x_pos_cor = np.cov(x_pos.T)
    x_neg_cor = np.cov(x_neg.T)

    x_cor_inverse = np.linalg.inv(x_pos_cor + x_neg_cor)
    print(x_cor_inverse.shape)

    x = np.concatenate((x_neg, x_pos), axis=0)
    w = np.dot(x_cor_inverse, (x_pos_mean - x_neg_mean).reshape((2, 1)))
    print('weight:', w)

    threshold = np.dot(w.reshape((1, 2)), (x_pos_mean + x_neg_mean).reshape((2, 1)))
    print('threshold', threshold)

    plt.figure(1)
    ax = plt.subplot(111)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    k = w[1] / w[0]
    x = np.mat(np.linspace(-9, 9))
    y = k * x
    plt.plot(x.T, y.T, c='g', ls='--')

    plt.plot(x_neg[:, 0], x_neg[:, 1], '+')
    plt.plot(x_pos[:, 0], x_pos[:, 1], '+')
    plt.savefig('./L4_' + str(1) + '.png')

    plt.figure(2)
    sns.distplot(np.dot(w.T, x_neg.T))
    sns.distplot(np.dot(w.T, x_pos.T))
    ax1 = plt.subplot(111)
    ax1.set_ylabel('density')
    plt.savefig('./L4_' + str(2) + '.png')


    plt.show()

if __name__ == '__main__':
    Fisher()


# output
# weight: [[-0.02092095]
#  [ 0.00012825]]
# threshold [[-0.00450561]]