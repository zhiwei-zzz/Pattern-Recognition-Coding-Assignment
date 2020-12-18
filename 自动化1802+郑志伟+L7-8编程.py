'''
@File  :自动化1802+郑志伟+L7-8编程.py
@Author:Zhiwei Zheng
@Date  :11/3/2020 11:59 PM
@Desc  :经纬度
        上海[121.43, 34.50]
        香港[114.10, 22.20]
        厦门[118.10, 24.46]
        南京[118.78, 32.05]
        海口[110.35, 20.02]
        杭州[120.20, 30.27]
        台北[121.56, 25.03]
        昆明[102.73, 25.04]
        成都[104.06, 30.67]
        乌鲁木齐[87.68, 43.77]

        横滨[139.39, 35.27]
        神户[135.10, 34.41]
        福冈[130.21, 33.39]
        长崎[129.51, 32.46]
        熊本[130.41, 32.47]
        鹿儿岛[130.33, 31.35]

        钓鱼岛[25.45， 123.18]
'''
from cvxopt import solvers, matrix
import numpy as np
import matplotlib.pyplot as plt


def main():
    raw_data = np.array([
            [1, 121.43, 34.50],
            [1, 114.10, 22.20],
            [1, 118.10, 24.46],
            [1, 118.78, 32.05],
            [1, 110.35, 20.02],
            [1, 120.20, 30.27],
            [1, 121.56, 25.03],

            # [1, 104.06, 30.67],
            # [1, 87.68, 43.77],
            # [1, 102.73, 25.04],

            [1, 139.39, 35.27],
            [1, 135.10, 34.41],
            [1, 130.21, 33.39],
            [1, 129.51, 32.46],
            [1, 130.41, 32.47],
            [1, 130.33, 31.35]
            ])

    data_gt = np.array([1,
               1,
               1,
               1,
               1,
               1,
               1,

               # 1,
               # 1,
               # 1,

               -1,
               -1,
               -1,
               -1,
               -1,
               -1])
    data = np.array(raw_data, copy=True)

    q = matrix(np.zeros((3, 1)))
    P = matrix(np.array([[0., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]]))
    for i in range(data.shape[0]):
        data[i] *= (data_gt[i] * (-1))

    G = matrix(data)
    h = matrix(np.ones((data.shape[0], 1)) * -1)
    sol = solvers.qp(P, q, G, h)
    w = sol['x']
    print(w)

    plt.figure(1)
    ax = plt.subplot(111)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # y = w0 + w1x1 + w2x2
    # k_line = -1 / (w[2] / w[1])
    y_line = np.arange(min(raw_data[:, 2]) - 0.5, max(raw_data[:, 2]) + 0.5, 0.2)
    x_line = -(w[0] / w[1]) - (w[2] * y_line / w[1])

    plt.plot(raw_data[0:7, 1], raw_data[0:7, 2], '+')
    plt.plot(raw_data[7:, 1], raw_data[7:, 2], '+')
    plt.plot(x_line, y_line, color='r', label="result")
    plt.savefig('./L7-8_' + str(1) + '.png')

    diaoyudao = np.array([1, 25.45, 123.18])
    result = np.dot(w.T, diaoyudao.T)
    print(result)
    if result > 0:
        print("钓鱼岛是中国的")
    else:
        print('算法错误')
    plt.show()


if __name__ == '__main__':
    main()
