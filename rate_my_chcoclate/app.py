import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        result = request.form
    cocoa = int(round(float(result['cocoa']),0))
    usa_company = result['usa']
    # usa_company = np.where(result['usa_company'] == 'Yes', 1, 0)
    new = pd.DataFrame({
        'Cocoa_Percent': cocoa,
        'Company_Location' : usa_company
    }, index = [0])
    prediction = pipe.predict(new)[0]
    # prediction = '${:,.2f}'.format(prediction)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug = True)
