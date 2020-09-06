import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
sc=load('std_scaler.bin')

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    prediction_text = ''
    txt_class = ''
    int_features = [int(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    print(final_features)
    if model.predict(sc.transform(final_features)) == [[4]]:
        prediction_text = 'Prone to Cancer'
        txt_class = 'prone-text'
    else:
        prediction_text = 'Not Prone to Cancer'
        txt_class = 'non-prone-text'

    return render_template('index.html', prediction_text=prediction_text,txt_class=txt_class)

if __name__ == "__main__":
    app.run(debug=True)


    