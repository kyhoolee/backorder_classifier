import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import random
import time
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve
import order_preprocess as o_pre


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
    f_score = (precision*recall) * 2.0 / (precision + recall)
    print(precision, recall, f_score)
    return (precision, recall)


def evalModel(clf, data):
    result = clf.predict(data[:, 1:22])
    print(result.shape)
    print(result)
    #countResult(data, result)
    truePos, trueNeg, falsePos, falseNeg = countResult(data, result)
    evalResult(truePos, trueNeg, falsePos, falseNeg)


def predict(data, clf):
    result = clf.predict(data[:, 1:22])
    return result


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


def train_write(train, file_path):
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


def sample_data(data, neg_prob):
    result = []
    for i in range(0, data.shape[0]):
        if data[i, -1] == 0:
            prob = random.uniform(0,1)
            if prob < neg_prob:
                result.append(data[i])
        else:
            result.append(data[i])

    return np.array(result)


def duplicate_data(data, pos_times):
    result = []
    for i in range(0, data.shape[0]):
        if data[i, -1] == 1:
            for t in range(0, pos_times):
                result.append(data[i])
        else:
            result.append(data[i])

    return np.array(result)


def plot_roc(valid, clf, model_name):
    # calculate the fpr and tpr for all thresholds of the classification
    probs = clf.predict_proba(valid[:, 1:22])
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(valid[:, -1], preds)
    roc_auc = metrics.auc(fpr, tpr)

    # roc
    import matplotlib.pyplot as plt
    plt.title('AUC ' + model_name)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # precision-recall-curve

    precision, recall, thresholds = precision_recall_curve(valid[:, -1], preds)
    area = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='Area Under Curve = %0.3f' % area)
    plt.legend(loc='lower left')
    plt.title('Precision-Recall ' + model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.show()


def training_from_csv_data(data_file, model_file):
    data = o_pre.read_data(data_file)
    train_write(data, model_file)


def build_classifier_sample():
    # write_data()
    train, valid = read_data('data/d_train_set.csv',
                             'data/d_valid_set.csv')

    # print(valid)
    print(valid[:, 1:-1].shape)
    countPosNeg(valid)
    print(train[:, 1:-1].shape)
    countPosNeg(train)

    # sample_train = sample_data(train, 0.1)
    # countPosNeg(sample_train)

    duplicate_train = duplicate_data(train, 10)
    countPosNeg(duplicate_train)
    train_write(duplicate_train, 'model/rf_d_duplicate_10.model')

def eval_result_sample(model_file, model_name):

    #write_data()
    train, valid = read_data('data/d_train_set.csv',
                             'data/d_valid_set.csv')

    #print(valid)
    print(valid[:, 1:-1].shape)
    countPosNeg(valid)
    print(train[:, 1:-1].shape)
    countPosNeg(train)


    clf = readModel(model_file)
    evalModel(clf, valid)
    plot_roc(valid, clf, model_name)

def eval_result_rf():
    eval_result_sample("model/rf_d_duplicate.model", "random forest")


if __name__ == "__main__":
    print()
    eval_result_rf()












