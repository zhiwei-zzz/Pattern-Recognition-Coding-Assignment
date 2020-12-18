'''
@File  :自动化1802+郑志伟+L10-11编程.py
@Author:Zhiwei Zheng
@Date  :2020/12/12 14:05
@Desc  :
'''

import torch
from torch import nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 5000
lr = 0.01
# Using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multi_train(model_list, epochs, lr_list, X_train, y_train, X_test, y_test):
    def model_train(model, criterion, optimizer, epochs, X_train, y_train, X_test, y_test):
        def model_eval(model, X_test, y_test):
            model.eval()

            output = model(X_test.float().to(device))
            loss = criterion(output, y_test.long().to(device))
            val_loss = loss.item()
            predicted = output.data
            val_correct = (torch.argmax(predicted.cpu(), axis=1) == y_test.long().cpu()).sum().item()
            val_total = y_test.size(0)

            print('val_loss: %.03f | val_acc: %.3f'
                  % (val_loss, val_correct / val_total))

        train_loss = []
        train_acc = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train.to(device))
            loss = criterion(output, y_train.long().to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            predicted = output.data
            train_correct = (torch.argmax(predicted.cpu(), axis=1) == y_train.long().cpu()).sum().item()
            train_total = y_train.size(0)
            train_acc.append(train_correct / train_total)

        if X_test is not None and y_test is not None:
            model_eval(model, X_test, y_test)

        return train_loss, train_acc

    loss_records = []
    acc_records = []

    for i in notebook.tqdm(range(len(model_list))):
        model = model_list[i]
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_list[i])
        train_loss, train_acc = model_train(model, criterion, optimizer, epochs, X_train, y_train, X_test, y_test)
        loss_records.append(train_loss)
        acc_records.append(train_acc)

    return loss_records, acc_records


def exp_net_depth(num_input, num_output, X_train, y_train, X_test=None, y_test=None):
    model1_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, num_output)
    )

    model2_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output)
    )

    model4_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output)
    )

    model8_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output)
    )

    model16_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output)
    )

    model32_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output)
    )

    model_list = [model1_16, model2_16, model4_16, model8_16, model16_16, model32_16]

    loss_records, acc_records = multi_train(model_list, epochs, [0.001] * 6, X_train, y_train, X_test, y_test)

    return loss_records, acc_records


def exp_hidden_neuron(num_input, num_output, X_train, y_train, X_test=None, y_test=None):
    model1_2 = nn.Sequential(
        nn.Linear(num_input, 2),
        nn.ReLU(),
        nn.Linear(2, num_output),
    )

    model1_4 = nn.Sequential(
        nn.Linear(num_input, 4),
        nn.ReLU(),
        nn.Linear(4, num_output),
    )

    model1_8 = nn.Sequential(
        nn.Linear(num_input, 8),
        nn.ReLU(),
        nn.Linear(8, num_output),
    )

    model1_16 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, num_output),
    )

    model1_32 = nn.Sequential(
        nn.Linear(num_input, 32),
        nn.ReLU(),
        nn.Linear(32, num_output),
    )

    model_list = [model1_2, model1_4, model1_8, model1_16, model1_32]
    loss_records, acc_records = multi_train(model_list, epochs, [0.001] * 5, X_train, y_train, X_test, y_test)

    return loss_records, acc_records


def exp_lr(num_input, num_output, X_train, y_train, X_test=None, y_test=None):
    model1 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output),
    )

    model2 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output),
    )

    model3 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output),
    )

    model4 = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output),
    )

    model_list = [model1, model2, model3, model4]

    loss_records, acc_records = multi_train(model_list, epochs, [0.0001, 0.001, 0.01, 0.1], X_train, y_train, X_test,
                                            y_test)

    return loss_records, acc_records


def exp_activation(num_input, num_output, X_train, y_train, X_test=None, y_test=None):
    model_relu = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_output),
    )

    model_tanh = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, num_output),
    )

    model_sigmoid = nn.Sequential(
        nn.Linear(num_input, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, num_output),
    )

    model_list = [model_relu, model_tanh, model_sigmoid]

    loss_records, acc_records = multi_train(model_list, epochs, [0.001] * 3, X_train, y_train, X_test, y_test)

    return loss_records, acc_records


def iris_data_generator(normalize=True):
    def data_normalize(X_train, X_test):
        sc = StandardScaler()
        sc.fit(X_train)
        _X_train = sc.transform(X_train)
        _X_test = sc.transform(X_test)

        return _X_train, _X_test

    iris = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])
    X = np.array(
        iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']], dtype=np.float32)
    y = iris[['Species']].copy()
    y[y['Species'] == 'setosa'] = 0
    y[y['Species'] == 'versicolor'] = 1
    y[y['Species'] == 'virginica'] = 2
    y = np.array(y).reshape(-1)

    X0_train, X0_test, y0_train, y0_test = train_test_split(X[np.where(y == 0)],
                                                            y[np.where(
                                                                y == 0)],
                                                            test_size=20, random_state=1)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X[np.where(y == 1)],
                                                            y[np.where(
                                                                y == 1)],
                                                            test_size=20, random_state=2)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X[np.where(y == 2)],
                                                            y[np.where(
                                                                y == 2)],
                                                            test_size=20, random_state=3)
    X_train = np.concatenate((X0_train, X1_train, X2_train), axis=0)
    y_train = np.concatenate((y0_train, y1_train, y2_train))
    X_test = np.concatenate((X0_test, X1_test, X2_test), axis=0)
    y_test = np.concatenate((y0_test, y1_test, y2_test))
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    if normalize == True:
        X_train, X_test = data_normalize(X_train, X_test)

    return torch.Tensor(X_train.astype(float)), torch.from_numpy(y_train.astype(float)), torch.from_numpy(X_test.astype(float)), torch.from_numpy(y_test.astype(float), )


# Problem 1
def Problem1():
    print("Working on Problem 1")
    X1 = torch.tensor([[3, 0.4],
                       [1, 1],
                       [3, 3],
                       [2, 0.5],
                       [3, 1],
                       [1, 3],
                       [1, 2],
                       [2, 2],
                       [3, 2]])
    y1 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int64)

    loss_records, acc_records = exp_net_depth(2, 3, X1, y1)
    fig1, ax = plt.subplots(4, 2, figsize=(14, 20))

    ax = ax.flatten()

    ax[0].plot(loss_records[0])
    ax[0].plot(loss_records[1])
    ax[0].plot(loss_records[2])
    ax[0].plot(loss_records[3])
    ax[0].plot(loss_records[4])
    ax[0].plot(loss_records[5])
    ax[0].legend(
        ('hidden_depth=1', 'hidden_depth=2', 'hidden_depth=4', 'hidden_depth=8', 'hidden_depth=16', 'hidden_depth=32'),
        loc='upper right')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')

    ax[1].plot(acc_records[0])
    ax[1].plot(acc_records[1])
    ax[1].plot(acc_records[2])
    ax[1].plot(acc_records[3])
    ax[1].plot(acc_records[4])
    ax[1].plot(acc_records[5])
    ax[1].legend(
        ('hidden_depth=1', 'hidden_depth=2', 'hidden_depth=4', 'hidden_depth=8', 'hidden_depth=16', 'hidden_depth=32'),
        loc='upper right')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Accuracy')

    loss_records, acc_records = exp_hidden_neuron(2, 3, X1, y1)

    ax[2].plot(loss_records[0])
    ax[2].plot(loss_records[1])
    ax[2].plot(loss_records[2])
    ax[2].plot(loss_records[3])
    ax[2].plot(loss_records[4])
    ax[2].legend(('neuron=1', 'neuron=2', 'neuron=4', 'neuron=8', 'neuron=16'), loc='upper right')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('Loss')

    ax[3].plot(acc_records[0])
    ax[3].plot(acc_records[1])
    ax[3].plot(acc_records[2])
    ax[3].plot(acc_records[3])
    ax[3].plot(acc_records[4])
    ax[3].legend(('neuron=1', 'neuron=2', 'neuron=4', 'neuron=8', 'neuron=16'), loc='upper right')
    ax[3].set_xlabel('Iterations')
    ax[3].set_ylabel('Accuracy')

    loss_records, acc_records = exp_lr(2, 3, X1, y1)

    ax[4].plot(loss_records[0])
    ax[4].plot(loss_records[1])
    ax[4].plot(loss_records[2])
    ax[4].plot(loss_records[3])
    ax[4].legend(('lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1'), loc='upper right')
    ax[4].set_xlabel('Iterations')
    ax[4].set_ylabel('Loss')

    ax[5].plot(acc_records[0])
    ax[5].plot(acc_records[1])
    ax[5].plot(acc_records[2])
    ax[5].plot(acc_records[3])
    ax[5].legend(('lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1'), loc='upper right')
    ax[5].set_xlabel('Iterations')
    ax[5].set_ylabel('Accuracy')

    loss_records, acc_records = exp_activation(2, 3, X1, y1)

    ax[6].plot(loss_records[0])
    ax[6].plot(loss_records[1])
    ax[6].plot(loss_records[2])
    ax[6].legend(('ReLU', 'Tanh', 'Sigmoid'), loc='upper right')
    ax[6].set_xlabel('Iterations')
    ax[6].set_ylabel('Loss')

    ax[7].plot(acc_records[0])
    ax[7].plot(acc_records[1])
    ax[7].plot(acc_records[2])
    ax[7].legend(('ReLU', 'Tanh', 'Sigmoid'), loc='upper right')
    ax[7].set_xlabel('Iterations')
    ax[7].set_ylabel('Accuracy')

    fig1.savefig('Problem1.PNG')


def Problem2():
    print("Working on Problem 2")
    X2 = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 1, 0],
                       [1, 1, 1, 0, 1, 0, 0, 1, 0],
                       [1, 0, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.float)
    y2 = torch.tensor([0, 1, 2], dtype=torch.int64)

    loss_records, acc_records = exp_net_depth(9, 3, X2, y2)

    fig2, ax = plt.subplots(4, 2, figsize=(14, 20))

    ax = ax.flatten()

    ax[0].plot(loss_records[0])
    ax[0].plot(loss_records[1])
    ax[0].plot(loss_records[2])
    ax[0].plot(loss_records[3])
    ax[0].plot(loss_records[4])
    ax[0].plot(loss_records[5])
    ax[0].legend(
        ('hidden_depth=1', 'hidden_depth=2', 'hidden_depth=4', 'hidden_depth=8', 'hidden_depth=16', 'hidden_depth=32'),
        loc='upper right')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')

    ax[1].plot(acc_records[0])
    ax[1].plot(acc_records[1])
    ax[1].plot(acc_records[2])
    ax[1].plot(acc_records[3])
    ax[1].plot(acc_records[4])
    ax[1].plot(acc_records[5])
    ax[1].legend(
        ('hidden_depth=1', 'hidden_depth=2', 'hidden_depth=4', 'hidden_depth=8', 'hidden_depth=16', 'hidden_depth=32'),
        loc='upper right')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Accuracy')

    loss_records, acc_records = exp_hidden_neuron(9, 3, X2, y2)

    ax[2].plot(loss_records[0])
    ax[2].plot(loss_records[1])
    ax[2].plot(loss_records[2])
    ax[2].plot(loss_records[3])
    ax[2].plot(loss_records[4])
    ax[2].legend(('neuron=1', 'neuron=2', 'neuron=4', 'neuron=8', 'neuron=16'), loc='upper right')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('Loss')

    ax[3].plot(acc_records[0])
    ax[3].plot(acc_records[1])
    ax[3].plot(acc_records[2])
    ax[3].plot(acc_records[3])
    ax[3].plot(acc_records[4])
    ax[3].legend(('neuron=1', 'neuron=2', 'neuron=4', 'neuron=8', 'neuron=16'), loc='upper right')
    ax[3].set_xlabel('Iterations')
    ax[3].set_ylabel('Accuracy')

    loss_records, acc_records = exp_lr(9, 3, X2, y2)

    ax[4].plot(loss_records[0])
    ax[4].plot(loss_records[1])
    ax[4].plot(loss_records[2])
    ax[4].plot(loss_records[3])
    ax[4].legend(('lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1'), loc='upper right')
    ax[4].set_xlabel('Iterations')
    ax[4].set_ylabel('Loss')

    ax[5].plot(acc_records[0])
    ax[5].plot(acc_records[1])
    ax[5].plot(acc_records[2])
    ax[5].plot(acc_records[3])
    ax[5].legend(('lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1'), loc='upper right')
    ax[5].set_xlabel('Iterations')
    ax[5].set_ylabel('Accuracy')

    loss_records, acc_records = exp_activation(9, 3, X2, y2)

    ax[6].plot(loss_records[0])
    ax[6].plot(loss_records[1])
    ax[6].plot(loss_records[2])
    ax[6].legend(('ReLU', 'Tanh', 'Sigmoid'), loc='upper right')
    ax[6].set_xlabel('Iterations')
    ax[6].set_ylabel('Loss')

    ax[7].plot(acc_records[0])
    ax[7].plot(acc_records[1])
    ax[7].plot(acc_records[2])
    ax[7].legend(('ReLU', 'Tanh', 'Sigmoid'), loc='upper right')
    ax[7].set_xlabel('Iterations')
    ax[7].set_ylabel('Accuracy')

    fig2.savefig('Problem2.PNG')


def Problem3():
    print("Working on Problem 3")
    X_train, y_train, X_test, y_test = iris_data_generator()

    loss_records, acc_records = exp_net_depth(4, 3, X_train, y_train, X_test, y_test)

    fig3, ax = plt.subplots(4, 2, figsize=(14, 20))

    ax = ax.flatten()

    ax[0].plot(loss_records[0])
    ax[0].plot(loss_records[1])
    ax[0].plot(loss_records[2])
    ax[0].plot(loss_records[3])
    ax[0].plot(loss_records[4])
    ax[0].plot(loss_records[5])
    ax[0].legend(
        ('hidden_depth=1', 'hidden_depth=2', 'hidden_depth=4', 'hidden_depth=8', 'hidden_depth=16', 'hidden_depth=32'),
        loc='upper right')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')

    ax[1].plot(acc_records[0])
    ax[1].plot(acc_records[1])
    ax[1].plot(acc_records[2])
    ax[1].plot(acc_records[3])
    ax[1].plot(acc_records[4])
    ax[1].plot(acc_records[5])
    ax[1].legend(
        ('hidden_depth=1', 'hidden_depth=2', 'hidden_depth=4', 'hidden_depth=8', 'hidden_depth=16', 'hidden_depth=32'),
        loc='upper right')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Accuracy')

    loss_records, acc_records = exp_hidden_neuron(4, 3, X_train, y_train, X_test, y_test)

    ax[2].plot(loss_records[0])
    ax[2].plot(loss_records[1])
    ax[2].plot(loss_records[2])
    ax[2].plot(loss_records[3])
    ax[2].plot(loss_records[4])
    ax[2].legend(('neuron=1', 'neuron=2', 'neuron=4', 'neuron=8', 'neuron=16'), loc='upper right')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('Loss')

    ax[3].plot(acc_records[0])
    ax[3].plot(acc_records[1])
    ax[3].plot(acc_records[2])
    ax[3].plot(acc_records[3])
    ax[3].plot(acc_records[4])
    ax[3].legend(('neuron=1', 'neuron=2', 'neuron=4', 'neuron=8', 'neuron=16'), loc='upper right')
    ax[3].set_xlabel('Iterations')
    ax[3].set_ylabel('Accuracy')

    loss_records, acc_records = exp_lr(4, 3, X_train, y_train, X_test, y_test)

    ax[4].plot(loss_records[0])
    ax[4].plot(loss_records[1])
    ax[4].plot(loss_records[2])
    ax[4].plot(loss_records[3])
    ax[4].legend(('lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1'), loc='upper right')
    ax[4].set_xlabel('Iterations')
    ax[4].set_ylabel('Loss')

    ax[5].plot(acc_records[0])
    ax[5].plot(acc_records[1])
    ax[5].plot(acc_records[2])
    ax[5].plot(acc_records[3])
    ax[5].legend(('lr=0.0001', 'lr=0.001', 'lr=0.01', 'lr=0.1'), loc='upper right')
    ax[5].set_xlabel('Iterations')
    ax[5].set_ylabel('Accuracy')

    loss_records, acc_records = exp_activation(4, 3, X_train, y_train, X_test, y_test)

    ax[6].plot(loss_records[0])
    ax[6].plot(loss_records[1])
    ax[6].plot(loss_records[2])
    ax[6].legend(('ReLU', 'Tanh', 'Sigmoid'), loc='upper right')
    ax[6].set_xlabel('Iterations')
    ax[6].set_ylabel('Loss')

    ax[7].plot(acc_records[0])
    ax[7].plot(acc_records[1])
    ax[7].plot(acc_records[2])
    ax[7].legend(('ReLU', 'Tanh', 'Sigmoid'), loc='upper right')
    ax[7].set_xlabel('Iterations')
    ax[7].set_ylabel('Accuracy')

    fig3.savefig("Problem3.PNG")


if __name__ == '__main__':
    # Problem1()
    # Problem2()
    Problem3()
