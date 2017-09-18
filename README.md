# Simple ML Model on AWS lambda Service

The purpose of this project is to show how to create a ML model and make it
available through a AWS lambda service.

The main requirements for this projects are:

- scikit-learn==0.19.0
- scipy==0.19.1
- Flask==0.12.2
- numpy==1.13.1
- pandas==0.20.3
- requests==2.18.4
- zappa==0.43.2
- awscli==1.11.153

You also need to have a valid AWS account from amazon.

## Preparing the environment

1. Creating the virtual environment:

  ```
virtualenv ml_cloud_aws
  ```

2. Activate the virtual environment:
  ```
source ml_cloud_aws/bin/activate
  ```
3. Clone the repository:
    ```
    cd ml_cloud_aws
    git clone https://github.com/dasoto/ml_cloud_aws.git
    ```

## Create your ML model
On this example we are using the dataset of breast cancer included in our scikit-learn.datasets.
Run the script to create the machine learning model model.pkl:

  ```
  python ml_cloud_src/model_creation.py
  ```
Now you have your model.pkl file that can be loaded as described in test_model.py to test your odel locally or app.py

## Load your model into a local web server using Flask
Now that you have your model, the app.py file contains a very basic example of a webservice that can answer POST requests. You can test it locally running:

1. Running the Flask server
  ```
  python app.py
  ```
The code is:
  ```python
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
  ```
2. Testing locally using curl:
TOu can test locally using the curl command in linux or mac:
  ```
  curl -d '[
        {  "1":1.62500000e+01,   "2":1.95100000e+01,   "3":1.09800000e+02,
           "4":8.15800000e+02,   "5":1.02600000e-01,   "6":1.89300000e-01,
           "7":2.23600000e-01,   "8":9.19400000e-02,   "9":2.15100000e-01,
           "10":6.57800000e-02,  "11":3.14700000e-01,   "12":9.85700000e-01,
           "13":3.07000000e+00,   "14":3.31200000e+01,   "15":9.19700000e-03,
           "16":5.47000000e-02,   "17":8.07900000e-02,   "18":2.21500000e-02,
           "19":2.77300000e-02,   "20":6.35500000e-03,   "21":1.73900000e+01,
           "22":2.30500000e+01,   "23":1.22100000e+02,   "24":9.39700000e+02,
           "25":1.37700000e-01,   "26":4.46200000e-01,   "27":5.89700000e-01,
           "28":1.77500000e-01,   "29":3.31800000e-01,   "30":9.13600000e-02}
  ]' -H "Content-Type: application/json" \
       -X POST http://127.0.0.1/predict && \
      echo -e "\n -> predict OK"
  ```
3. Test using a python script:
  ```python
  import requests
  import json

  url = 'http://127.0.0.1:8080/predict'
  data = json.dumps([{  "1":1.62500000e+01,   "2":1.95100000e+01,   "3":1.09800000e+02,
           "4":8.15800000e+02,   "5":1.02600000e-01,   "6":1.89300000e-01,
           "7":2.23600000e-01,   "8":9.19400000e-02,   "9":2.15100000e-01,
           "10":6.57800000e-02,  "11":3.14700000e-01,   "12":9.85700000e-01,
           "13":3.07000000e+00,   "14":3.31200000e+01,   "15":9.19700000e-03,
           "16":5.47000000e-02,   "17":8.07900000e-02,   "18":2.21500000e-02,
           "19":2.77300000e-02,   "20":6.35500000e-03,   "21":1.73900000e+01,
           "22":2.30500000e+01,   "23":1.22100000e+02,   "24":9.39700000e+02,
           "25":1.37700000e-01,   "26":4.46200000e-01,   "27":5.89700000e-01,
           "28":1.77500000e-01,   "29":3.31800000e-01,   "30":9.13600000e-02}])
  r = requests.post(url,data)
  print(r.json())
  ```

## Upload to Amazon AWS Lambda service
To complete this step we will use awscli and zappa.
1. Install awscli:
  ```
  pip install awscli
  ```
2. Install zappa:
  ```
  pip install zappa
  ```
3. Configure amazon credentials and region:
  ```
  aws configure
  ```
4. Init zappa:
  ```
  zappa init
  ```
5. Edit your zappa settings file zappa_settings.json and add the line slim_handler:
  ```json
  {
      "dev": {
          "app_function": "app.app",
          "aws_region": "us-east-1",
          "profile_name": "default",
          "s3_bucket": "zappa-ml-model-dev",
          "slim_handler": true
      }
  }
  ```
6. Deploy your webservice:
  ```
  zappa deploy
  ```
7. With the address that will provide the previous step you can modify the python script or curl command to test your service!!!
