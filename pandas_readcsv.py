import pandas as pd
import numpy as np

file_path = "/home/kylee/tutorial/learn/projects/coding_skills/sendo_test/dataset/sample_set.csv"
backorders = pd.read_csv(file_path)
#print(backorders)

npOrders = backorders.as_matrix()
#print(npOrders)

np.random.shuffle(npOrders)
train, valid = npOrders[:80, :], npOrders[80:, :]



