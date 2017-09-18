from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

print('Loading dataset')
data = load_breast_cancer()
print('Separating training set and test set.')
X_train = data.data[:-100]
y_train = data.target[:-100]
X_test = data.data[-100:]
y_test = data.target[-100:]

print('Training our model using KNeighbors')

clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Creating model file model.pkl')
joblib.dump(clf, 'model.pkl')
