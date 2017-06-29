from sklearn import svm
from sklearn import datasets
import pickle

clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data, iris.target
clf.fit(X,y)


s = pickle.dumps(clf)
#print(s)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])



