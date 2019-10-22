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
    lot_area = result['lot_area']
    abvgr_area = result['abvgr_area']
    floor1_area = result['floor1_area']
    new = pd.DataFrame({
        'Lot Area' : lot_area,
        'Gr Liv Area' : abvgr_area,
        '1st Flr SF' : floor1_area
    }, index = [0])
    prediction = pipe.predict(new)[0]
    prediction = int(prediction//1)
    prediction = '${:,.2f}'.format(prediction)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug = True)
