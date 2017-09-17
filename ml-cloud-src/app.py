from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

print('Loading Model...')
clf = joblib.load('model.pkl')
print(clf)
print('Model loaded.')
@app.route('/predict', methods=['POST'])
def predict():
    print('Receiving data of prediction')
    data = request.get_json(force = True)
    data = dict(data[0])
    print(data)
    print('Printing data 0')
    print(data['1'])
    predict_request = [data['1'],data['2'],data['3'],data['4'],data['5'],data['6'],data['7'],\
                       data['8'],data['9'],data['10'],data['11'],data['12'],data['13'],\
                       data['14'],data['15'],data['16'],data['17'],data['18'],data['19'],\
                       data['20'],data['21'],data['22'],data['23'],data['24'],data['25'],\
                       data['26'],data['27'],data['28'],data['29'],data['30']]
    print('After...')
    print(predict_request)
    predict_request = np.array(predict_request).reshape(1,-1)
    prediction = str(clf.predict(predict_request).astype(int))
    print('Predicted value is:')
    print(prediction)
    print(type(prediction))
    return jsonify({'prediction': list(prediction)})
    #return jsonify(result = prediction)
if __name__ == '__main__':
    print('Passing main:')
    app.run(port=8080,debug=True)
