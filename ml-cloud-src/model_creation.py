from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

data = load_breast_cancer()
X_train = data.data[:-100]
y_train = data.target[:-100]
X_test = data.data[-100:]
y_test = data.target[-100:]

clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

joblib.dump(clf, 'model.pkl')
