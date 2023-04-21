from flask import Flask,render_template,redirect,flash,request
from flask_cors import CORS, cross_origin
import pickle
import numpy as np

app=Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    family_history = request.form.get('family_history')
    benefits = request.form.get('benefits')
    care_options = request.form.get('care_options')
    anonymity = request.form.get('anonymity')
    leave = request.form.get('leave')
    work_interfere = request.form.get('work_interfere')

    # Reshape input into a 2D array
    input_data = np.array(
        [age, gender, family_history, benefits, care_options, anonymity, leave, work_interfere]).reshape(1, -1)

    # Load the model
    with open("C:/Users/GEETHESHWAR/Downloads/boostmodel.pkl", 'rb') as f:
        model = pickle.load(f)

    # Make prediction and return result
    prediction = model.predict(input_data)
    yes="Person has mental illness"
    no='person is mentally stable'
    return "{}".format(yes if str(prediction[0])=='1' else no)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=1234, debug=True)
