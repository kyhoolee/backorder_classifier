import pandas as pd
import numpy as np
import order_classifier
import random


def read_data(file_path):
    print()
    backorders = pd.read_csv(file_path)
    # print(backorders)

    npOrders = backorders.as_matrix()
    #print(npOrders)
    data = convert_instance(npOrders)
    return data


def convert_boolean(value):
    if value == 'No':
        return 0
    elif value == 'Yes':
        return 1
    else:
        return -99


def convert_instance(x):
    print()
    for row in x:
        #print(row)
        if np.isnan(row[2]):
            row[2] = -99
        row[12] = convert_boolean(row[12])
        for i in range(17, 23):
            row[i] = convert_boolean(row[i])

        #print(row)

    return x


def divide_dataset(data, train_prob):
    print()
    train = []
    valid = []
    for i in range(0, data.shape[0]):
        prob = random.uniform(0,1)
        if prob < train_prob:
            train.append(data[i])
        else:
            valid.append(data[i])
    train = np.array(train)
    valid = np.array(valid)
    return (train, valid)


def write_datafile(data, file_path):
    #data.tofile(file_path, sep=",", format='%10.5f')
    df = pd.DataFrame(data)
    df.to_csv(file_path, header=None, index=False)


def write_dataset(train, valid, train_file, valid_file):
    write_datafile(train, train_file)
    write_datafile(valid, valid_file)


def pre_process():
    data = read_data("data/training_set.csv")
    #print(data)

    print(data.shape)
    train, valid = divide_dataset(data, 0.8)
    print(train.shape)
    print(valid.shape)
    write_dataset(train, valid, "data/d_train_set.csv", "data/d_valid_set.csv")



if __name__ == "__main__":
    #pre_process()
    data = read_data("data/training_set.csv")
    order_classifier.countPosNeg(data)







