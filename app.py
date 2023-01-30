import numpy as np
from flask import Flask, request, render_template
import pickle
from pydantic import BaseModel
from flask_pydantic import validate
import pandas as pd

app = Flask(__name__)

with open('models/regressor1.pkl','rb') as f:
    model = pickle.load(f)

class QueryParams(BaseModel):
    CleaningNo: float
    Total: float
    CycleNo: float
    AvgFlowrate: float
    AvgFeedCond: float
    AvgCharCur: float
    AvgDisCur: float

@app.route('/apipredict', methods=['POST','GET'])
@validate()
def mcdi_endpoint(body:QueryParams):
    df = pd.DataFrame([body.dict().values()], columns=body.dict().keys())
    pred = model.predict(df)
    return {"Cycle Time": int(pred)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    return render_template('index.html', prediction_text = "Cycle Time: {}".format(int(prediction)))

if __name__ == "__main__":
    app.run()