import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import random
import time


def readDataset(train_prob):
    file_path = "/home/kylee/tutorial/learn/projects/coding_skills/sendo_test/dataset/sample_set.csv"
    backorders = pd.read_csv(file_path)
    #print(backorders)

    npOrders = backorders.as_matrix()
    #print(npOrders)

    np.random.shuffle(npOrders)
    total = npOrders.shape[0]
    train_size = int(total * train_prob)
    train, valid = npOrders[:train_size, :], npOrders[train_size:, :]

    return (train, valid)


def trainSvmModel(train):
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train[:,1:-1], train[:,-1])
    return clf


def trainMlpModel(train, size):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(size,), random_state=1)
    clf.fit(train[:, 1:-1], train[:, -1])
    return clf

def trainRFModel(train):
    clf = RandomForestClassifier(n_jobs = -1)
    clf.fit(train[:, 1:-1], train[:, -1])
    return clf


def validModel(valid, clf):
    result = clf.fit(valid[:,1:-1])
    return result


def countPosNeg(data):
    pos = 0
    neg = 0

    for row in data:
        if(row[-1] == 0):
            neg += 1
        else:
            pos += 1

    print(pos, neg)

    return (pos, neg)


def countResult(data, result):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    for i in range(0, result.shape[0]) :
        if data[i, -1] == 1 and result[i] == 1:
            truePos += 1
        elif data[i, -1] == 1 and result[i] == 0:
            falseNeg += 1
        elif data[i, -1] == 0 and result[i] == 1:
            falsePos += 1
        elif data[i, -1] == 0 and result[i] == 0:
            trueNeg += 1

    print(truePos, trueNeg, falsePos, falseNeg)

    return(truePos, trueNeg, falsePos, falseNeg)

def evalResult(truePos, trueNeg, falsePos, falseNeg):
    precision = truePos * 1.0 / (truePos + falsePos)
    recall = truePos * 1.0 / (truePos + falseNeg)
    print(precision, recall)
    return (precision, recall)

def evalModel(clf, data):
    result = clf.predict(data[:, 1:-1])
    print(result.shape)
    print(result)
    #countResult(data, result)
    truePos, trueNeg, falsePos, falseNeg = countResult(data, result)
    #evalResult(truePos, trueNeg, falsePos, falseNeg)


def writeModel(clf, file_path):
    joblib.dump(clf, file_path)

def readModel(file_path):
    clf = joblib.load(file_path)
    return clf


def writeNumpy(data, file_path):
    df = pd.DataFrame(data=data)
    df.to_csv(file_path, sep=',', header=False, index=False)

def writeData(train, valid, train_path, valid_path):
    writeNumpy(train, train_path)
    writeNumpy(valid, valid_path)


def train_write(train, file_path, size):
    start_time = time.time()
    clf = trainRFModel(train)
        #trainSvmModel(train)
    #trainMlpModel(train, size)
    total_time = time.time() - start_time
    print(total_time)

    writeModel(clf, file_path)

def write_data():
    (train, valid) = readDataset(0.1)
    print(train.shape)
    countPosNeg(train)

    print(valid.shape)
    countPosNeg(valid)

    writeData(train, valid, 'backorder_train.txt', 'backorder_valid.txt')

def readNumpy(file_path):
    data = pd.read_csv(file_path)
    dataNp = data.as_matrix()
    return dataNp


def read_data(train_path, valid_path):
    train = readNumpy(train_path)
    valid = readNumpy(valid_path)
    return (train, valid)






if __name__ == "__main__":
    print()
    #write_data()
    train, valid = read_data('filtered/duplicate_total_train.csv',
                             'filtered/total_valid.csv')

    #print(valid)
    print(valid[:, 1:-1].shape)
    countPosNeg(valid)
    print(train[:, 1:-1].shape)
    countPosNeg(train)

    #train_write(train, 'model/rf_duplicate_total.model', 50)
    clf = readModel('model/rf_duplicate_total.model')
    evalModel(clf, valid)

    #(2932, 131835, 2621, 6118)

    #(9050, 134456)

    #(2243, 334624)
    #(160, 332834, 1790, 2083)
    #(751, 328221, 6403, 1492)

    #2500s
    #(508, 330052, 4572, 1735)

    #3570.64846897
    #(913, 331345, 3279, 1330)

    #(1354, 331149, 3475, 889)
    #(881, 334162, 462, 1362)









