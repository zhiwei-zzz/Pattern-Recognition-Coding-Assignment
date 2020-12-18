'''
@File  :自动化1802+郑志伟+L9编程.py
@Author:Zhiwei Zheng
@Date  :11/4/2020 11:39 AM
@Desc  :
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import random


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


def read_data():
    with open('iris.csv', 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        f.close()

    setosa = []
    versicolor = []
    virginica = []

    for i in range(len(result) - 1):
        if result[i + 1][5] == 'setosa':
            setosa.append([result[i + 1][1], result[i + 1][2], result[i + 1][3], result[i + 1][4]])
        if result[i + 1][5] == 'versicolor':
            versicolor.append([result[i + 1][1], result[i + 1][2], result[i + 1][3], result[i + 1][4]])
        if result[i + 1][5] == 'virginica':
            virginica.append([result[i + 1][1], result[i + 1][2], result[i + 1][3], result[i + 1][4]])

    return np.array(setosa, dtype='float32'), np.array(versicolor, dtype='float32'), np.array(virginica,
                                                                                              dtype='float32')


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

    for i in range(2000):
        random_order = random.sample(range(x_num), x_num)  # randomly choosing data
        flag = True

        for j in random_order:  # exploit all dataset

            compare = np.sign(np.dot(w.T, input_data[j])) != input_gt[j]
            if flag is True and compare:  # judging the result
                flag = False

            w = (w + (compare * input_gt[j] * input_data[j].T))

        if flag is True:
            break

    precision = evaluate_w(w, input_data, input_gt, precision)
    # print(precision)
    return w, precision


def train(data_1, data_2, w_ini):
    data_concat = np.concatenate((data_1, data_2), axis=0)
    data_2_gt = np.ones(len(data_2)) * -1
    data_1_gt = np.ones(len(data_1))
    data_gt = np.concatenate((data_1_gt, data_2_gt), axis=0)
    w, _ = Perce(data_concat, data_gt, w_ini)

    return w


def ovo(setosa_test, versicolor_test, virginica_test, setosa_train, versicolor_train, virginica_train):
    print('ovo:')
    w_ini = [0, 0, 0, 0]
    se_vi_w = train(setosa_train, virginica_train, w_ini)
    se_ve_w = train(setosa_train, versicolor_train, w_ini)
    vi_ve_w = train(virginica_train, versicolor_train, w_ini)
    print(se_vi_w, '\n', se_ve_w, '\n', vi_ve_w)

    test_data = np.concatenate((setosa_test, versicolor_test, virginica_test))
    one_vector = np.ones((1, 60))
    test_data = np.c_[one_vector.T, test_data]
    w = np.concatenate(([se_vi_w], [se_ve_w], [vi_ve_w]))

    result = np.sign(np.dot(test_data, w.T))
    classify_result = []
    for i in range(60):
        se_vote = 0
        ve_vote = 0
        vi_vote = 0

        if result[i][0] > 0:
            se_vote += 1
        else:
            vi_vote += 1

        if result[i][1] > 0:
            se_vote += 1
        else:
            ve_vote += 1

        if result[i][2] > 0:
            vi_vote += 1
        else:
            ve_vote += 1
        vote = [se_vote, ve_vote, vi_vote]
        if se_vote == vi_vote and vi_vote == ve_vote:
            classify_result.append(-1)
        else:
            classify_result.append(vote.index(max(vote)))
    setosa_count = versicolor_count = virginica_count = 0
    for i in range(20):
        if classify_result[i] == 0:
            setosa_count += 1
        if classify_result[i + 20] == 1:
            versicolor_count += 1
        if classify_result[i + 40] == 2:
            virginica_count += 1
    print('ovo precision: setosa test precision: %.2f, versicolor test precision: %.2f, virginica test precision: '
          '%.2f, model presion: %.2f ' % (setosa_count / 20, versicolor_count / 20, virginica_count / 20,
                                          (setosa_count + versicolor_count + virginica_count) / 60))





def softmax(setosa_test, versicolor_test, virginica_test, setosa_train, versicolor_train, virginica_train, epoch,
            lr=0.01):
    print('\n', 'softmax:')
    w = np.zeros((3, 5))
    one_vector = np.ones((1, 90))
    train_data = np.concatenate((setosa_train, versicolor_train, virginica_train))
    train_data = np.c_[one_vector.T, train_data]
    train_label = np.zeros((90, 3))
    for i in range(30):
        train_label[i][0] = 1  # setosa
        train_label[i + 30][1] = 1  # versicolor
        train_label[i + 60][2] = 1  # virginica

    for i in range(epoch):
        s = np.dot(train_data, w.T)
        y = np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)
        gradient = np.dot((y - train_label).T, train_data)
        w = w - lr * gradient

    print(w)
    one_vector = np.ones((1, 60))
    test_data = np.concatenate((setosa_test, versicolor_test, virginica_test))
    test_data = np.c_[one_vector.T, test_data]
    s = np.dot(test_data, w.T)
    y = np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True)
    result = np.argmax(y, axis=1)
    setosa_count = 0
    versicolor_count = 0
    virginica_count = 0
    for i in range(20):
        if result[i] == 0:
            setosa_count += 1
        if result[i + 20] == 1:
            versicolor_count += 1
        if result[i + 40] == 2:
            virginica_count += 1
    print('softmax precision: setosa test precision: %.2f, versicolor test precision: %.2f, virginica test precision: '
          '%.2f, model presion: %.2f ' % (setosa_count / 20, versicolor_count / 20, virginica_count / 20,
                                          (setosa_count + versicolor_count + virginica_count) / 60))


if __name__ == '__main__':
    setosa_test = []
    versicolor_test = []
    virginica_test = []
    setosa_train = []
    versicolor_train = []
    virginica_train = []
    random_order = random.sample(range(50), 30)
    setosa, versicolor, virginica = read_data()
    for i in range(50):
        if i in random_order:
            setosa_train.append(setosa[i])
            virginica_train.append(virginica[i])
            versicolor_train.append(versicolor[i])
        else:
            setosa_test.append(setosa[i])
            virginica_test.append(virginica[i])
            versicolor_test.append(versicolor[i])

    epoch = 400
    ovo(setosa_test, versicolor_test, virginica_test, setosa_train, versicolor_train, virginica_train)
    softmax(setosa_test, versicolor_test, virginica_test, setosa_train, versicolor_train, virginica_train, epoch,
            lr=0.01)
