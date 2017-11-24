import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import random as rd
from cvxpy import *

style.use('ggplot')


class SVM():
    def fit(self, data, C=5):

        # train to get w and b for hyperplane
        # solving optimizational problem for min 1/2 ||w||^2 + C Sum(e_i)
        # subject to y_i (w^t x_i + b) >= 1 - e_i and e_i >= 0 for i = 1, ... , len data[0]

        n = len(data)
        Q = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                Q[i][j] = np.dot(data[i][:-1], data[j][:-1]) * data[i][-1] * data[j][-1]
                Q[j][i] = np.dot(data[i][:-1], data[j][:-1]) * data[i][-1] * data[j][-1]

        for vrstica in Q:
            print(vrstica)

        e = [1] * n
        a = Variable(n)

        y = [0] * n
        for i in range(n):
            y[i] = data[i][-1]

        obj = Minimize(1 / 2 * quad_form(a, Q) - a.T * e)

        constraints = [a.T * y == 0, a >= 0, a <= C]

        prob = Problem(obj, constraints)
        prob.solve()
        print("a je: ")
        print(a)
        a.get_data()
        return a

    def prepare_data(self, data, k=10):
        '''
        :param data:
        :param k:
        :return:
        '''
        data = rd.shuffle(data)

        split_data = partition(data, k)
        self.data_for_cross_validtion = []
        for i in range(k):
            test_data = []
            train_data = []
            for j in range(k):
                if j == i:
                    test_data = split_data[i]
                else:
                    train_data.append(split_data[j])
            train_data = [x for y in train_data for x in y]
            self.data_for_cross_validation.append((train_data, test_data))

    def normalize(self, data):
        '''
        :param data: data as [train_data, test_data], both array
        :return: Normalized data, so that all data is on interval [-1, 1]
        '''
        train_data = data[0]
        test_data = data[1]
        x = [0] * len(train_data[0])
        for instance in train_data:
            for pos, feature in enumerate(instance):
                if abs(feature) >= abs(x[pos]):
                    x[pos] = abs(feature)
        for instance in train_data:
            for pos, feature in enumerate(instance):
                feature = feature / x[pos]
        for instance in test_data:
            for pos, feature in enumerate(instance):
                feature = feature / x[pos]
        return (train_data, test_data)

    def set_class(self, data):
        '''
        :param data:
        :return: Data which predict class is either -1 or 1
        '''
        predictions = set()
        for example in data:
            for instance in example:
                predictions.add(instance[-1])
        mapping = {}
        inverse_mapping = {}
        predictions = list(predictions)
        mapping[predictions[0]] = 1
        inverse_mapping[1] = predictions[0]
        mapping[predictions[1]] = -1
        inverse_mapping[0] = predictions[1]
        for example in data:
            for instance in example:
                instance[-1] = mapping[instance[-1]]
        return data




def partition(lst, n):
    '''
    :param lst: array of elemnts
    :param n: number of partitions
    :return: array partitioned into n arrays
    '''
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]




data = [[1, 7, -1],
        [2, 8, -1],
        [3, 8, -1],
        [5, 1, 1],
        [6, -1, 1],
        [7, 3, 1],
        ]


print(data)
clf = SVM()

b = clf.fit(data)

