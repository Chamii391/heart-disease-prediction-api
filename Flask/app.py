from flask import Flask, render_template,request
import joblib
import numpy as np
from keras.models import load_model
from keras import backend as K

model = load_model('models/model-068.keras')

x_scaler = joblib.load('models/x_scaler.joblib')
y_scaler = joblib.load('models/y_scaler.joblib')

app = Flask(__name__)


@app.route('/')
def test():
    return render_template('test.html')


@app.route('/getresults', methods=['POST'])
def getresults():

    results = request.form

    print(results)

    name = results['name']
    gender = int(results['gender'])
    age = int(results['age'])
    tc = int(results['tc'])
    hdl = int(results['hdl'])
    smoke = int(results['smoke'])
    bpmed = int(results['bpmed'])
    diab = int(results['diab'])

    test_data = np.array([[gender, age, tc, hdl, smoke, bpmed, diab]])
    test_data = x_scaler.transform(test_data)

    prediction = model.predict(test_data)

    prediction = y_scaler.inverse_transform(prediction)

    resultsDict = {'name': name, 'prediction': prediction[0][0]}

    return render_template('results.html', resultsDict=resultsDict)



 

app.run(debug=True)