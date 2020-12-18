'''
@File  :lecture2.py
@Author:Zhiwei Zheng
@Date  :2020/10/26 11:07
@Desc  :
'''

import numpy as np
import random
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import warnings


# Problem 1
def Perce(input_data, input_gt, w_ini):
    '''

    Args:
        input_data: [N*d]matrics without x0
        input_gt: [N*1]vector
        w_ini: [d*1]vector  [w1, w2, w3 ...wd]

    Returns:
        w: [d*1]vector [w0, w1, w2, w3 ...wd]
        precision: [3*1]

    '''
    x_num = input_data.shape[0]
    w = w_ini

    one_vector = np.ones((1, x_num))
    input_data = np.c_[one_vector.T, input_data]  # considering w0 and x0 making x0 = 1
    w = np.insert(w, 0, [0])  # making w0 = 0
    precision = [0, 0, 0]
    flag_count = 0

    for i in range(25):
        random_order = random.sample(range(x_num), x_num)  # randomly choosing data
        flag = True

        for j in random_order:  # exploit all dataset

            compare = np.sign(np.dot(w.T, input_data[j])) != input_gt[j]
            if flag is True and compare:  # judging the result
                flag = False

            w_temp = (w + compare * input_gt[j] * input_data[j].T) / 2
            evaluate_result = evaluate(w_temp, input_data, input_gt)
            if evaluate_result > precision[2]:
                precision[2] = evaluate_result
                w = w_temp
                flag_count = 0
            else:
                flag_count += 1
                continue
        if flag is True or flag_count == 2 * len(input_data):
            break
    precision = evaluate_w(w, input_data, input_gt, precision)
    return w, precision


# testing code for function Perce()
# a = np.array([[0, 1],
#               [1, 1],
#               [100, 1],
#               [101, 1]])
# b = np.array([1,
#               1,
#               -1,
#               -1])
# w = np.array([0,
#               0])
#
# w, precision = Perce(a, b, w)
# print(w)

# Problem 2

def evaluate_w(w_weight, input_data, input_gt, precision):
    label_0 = label_1 = 0
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


def evaluate(w_weight, input_data, input_gt):
    count = 0
    for j in range(len(input_data)):
        if np.sign(np.dot(w_weight.T, input_data[j])) == input_gt[j]:
            count += 1
    return count / len(input_data)


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


def perception(x1_mean, x2_mean, y1_mean, y2_mean, figure_num):
    '''

    Args:
        x1_mean:
        x2_mean:
        y1_mean:
        y2_mean:

    Returns:

    '''
    x_neg, x_pos = dataset_generator(x1_mean, x2_mean, y1_mean, y2_mean)
    x_neg_gt = np.ones(len(x_neg)) * -1
    x_pos_gt = np.ones(len(x_pos))
    x_gt = np.concatenate((x_neg_gt, x_pos_gt), axis=0)
    x = np.concatenate((x_neg, x_pos), axis=0)
    w_ini = [0, 4]
    w, Precision = Perce(x, x_gt, w_ini)
    print("The weights of the model trained on Problem%d dataset is " % figure_num, w, )
    print("The precision of label'-1', label'1', and the whole dataset is {:.2%} {:.2%} {:.2%}.".format(Precision[0],
                                                                                                        Precision[1],
                                                                                                        Precision[2]))

    plt.figure(figure_num, figsize=(8, 5), dpi=80)
    ax = plt.subplot(111)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # y = w0 + w1x1 + w2x2
    # k_line = -1 / (w[2] / w[1])
    y_line = np.arange(min(x[:, 1]) - 0.5, max(x[:, 1]) + 0.5, 0.2)
    x_line = -(w[0] / w[1]) - (w[2] * y_line / w[1])

    plt.plot(x_neg[:, 0], x_neg[:, 1], '+')
    plt.plot(x_pos[:, 0], x_pos[:, 1], '+')
    plt.plot(x_line, y_line, color='r', label="result")
    plt.savefig('./L2_' + str(figure_num) + '.png')
    plt.show()


if __name__ == '__main__':
    perception(-5, 0, 5, 0, 2)  # [-5、-2、-1,0]对应 label 为-1
    perception(-2, 0, 2, 0, 3)
    perception(-1, 0, 1, 0, 4)
#
# Problem 5
# 在讨论2-4题实验结果前，先讨论函数的具体实现方法。因在实验中发现，假如说不给予任何限定，任由模型每错误分类一个样本即允许更改W权值，对于整体来说有可能
# 会为了学习那一个样本而降低了整个模型的准确度，在2-4的实验中该现象体现的最为明显。因此在本代码中，在每次进行一次W权值更新后即会对该W值进行一次对整个
# 数据样本的总体准确度。只在总体准确度上升的情况下，承认该新的W值，否则予以抛弃。（对于这个判断标准是存疑的，不知如此是否会更好，有可能分别对label'1'
# 与 label'2'与总体的准确度进行一个权值相加作为判断标准效果会更好，此处不予以深究。）该模型训练时，将400个数据集以一个整体为一次循环，最多循环25次，
# 以满足题目最多迭代10000次的限制。每次进行一个整个数据集的循环训练时，若某次循环结果准确度为100%，则提前结束训练并输出结果；若在训练中，连续学习800
# 次结果后总准确度并未增加，则也提前结束训练。（设置为800次是为了保证在两次循环中一定会遍历完所有的数据点，避免若设置为400次后，因数据为随机读取400次
# 数据而并没有真正实现便利完所有的数据而造成误判）。
# 对于2-4的实验结果，是具有一定随机性.在多次跑完实验后,每次总的分类准确度并不高，一般在85%左右。实验2-4的最大的感受就是对于W的更新需要加以限制，否则
# 结果反而有可能随着训练次数的增加而下降，在前文中也已详细描述。此外感知器算法在算法层面上就以注定了无法很好的分类这种相似度接近的两种类型。需要更加好的
# 算法进行代替。

# output
# The weights of the model trained on Problem2 dataset is  [0.5        1.43500708 0.10024659]
# The precision of label'-1', label'1', and the whole dataset is 100.00% 100.00% 100.00%.
# The weights of the model trained on Problem3 dataset is  [-0.1875      0.68018585  0.21957133]
# The precision of label'-1', label'1', and the whole dataset is 99.50% 95.00% 97.25%.
# The weights of the model trained on Problem4 dataset is  [-0.25        1.02461915 -0.17361609]
# The precision of label'-1', label'1', and the whole dataset is 90.50% 76.00% 83.25%.
