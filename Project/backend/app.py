from flask import Flask, request, jsonify
from flask_cors import CORS
import glob
import pandas as pd
import os

from api import getRequiredColumns, LSTMAlgorithm, getPredictonsFromModel, getManualPredictionForModel

app = Flask("Stock Price Prediction")
CORS(app)

df = None
cols, dateColName, closeColName = None, None, None
train_size = 0.75
totalEpochs = 2

session = {
    "training": {
        "status": "ready",
        "fileUploaded": False,
        "fileName": None,
        "totalEpochs": totalEpochs
    },
    "prediction": {
        "status": "ready",
        "preTrainedModelNames": None
    }
}

def updateEpochs(epoch):
    global session
    session['training']['epochs'] = epoch + 1

@app.route("/")
def index():
    return "Welcome to Stock Price Prediction API"

@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        global session, df, cols, dateColName, closeColName

        df = pd.read_csv(request.files['file'])
        df = df.dropna()
        
        cols, dateColName, closeColName = getRequiredColumns(df)
        
        dfColVals = []
        dfDateVals = []
        dfCloseVals = []

        for row in df[[dateColName] + cols].values:
            dfColVals.append(list(row))
            dfCloseVals.append(row[4])
            dfDateVals.append(row[0])

        session['training']['fileUploaded'] = True
        session['training']['fileName'] = request.files['file'].filename[:-4]
        session['training']['cols'] = [dateColName] + cols
        session['training']['dfColVals'] = dfColVals
        session['training']['dfCloseVals'] = dfCloseVals
        session['training']['dfDateVals'] = dfDateVals
        
        return jsonify(session['training'])
    else:
        return "This API accepts only POST requests"

@app.route("/startTraining", methods=['POST', 'GET'])
def startTraining():
    if request.method == "POST":
        global session, df

        fileName = request.form['fileName']

        os.makedirs("/opt/render/project/src/datasets", exist_ok=True)  # Updated path
        df.to_csv(f'/opt/render/project/src/datasets/{fileName}.csv', index=False)

        session['training']['status'] = "training"
        session['training']['epochs'] = 0

        model = LSTMAlgorithm(fileName, train_size, totalEpochs, updateEpochs=updateEpochs)

        session['training']['status'] = "trainingCompleted"
        return jsonify(session['training'])
    else:
        return "This API accepts only POST requests"

@app.route("/trainingStatus", methods=['POST', 'GET'])
def trainingStatus():
    if request.method == "POST":
        return jsonify(session['training'])
    else:
        return "This API accepts only POST requests"

@app.route("/getPreTrainedModels", methods=['POST', 'GET'])
def getPreTrainedModels():
    if request.method == "POST":
        global session

        os.makedirs("/opt/render/project/src/pretrained", exist_ok=True)  # Updated path
        files = glob.glob("/opt/render/project/src/pretrained/*.H5")
        files = [f.split("/")[-1][:-3] for f in files]

        session['prediction']['preTrainedModelNames'] = files
        return jsonify(session['prediction'])
    else:
        return "This API accepts only POST requests"

@app.route("/getPredictions", methods=['POST', 'GET'])
def getPredictions():
    if request.method == "POST":
        global session

        modelName = request.form['modelName']
        session['prediction']['modelName'] = modelName

        modelData = getPredictonsFromModel(modelName, train_size)
        session['prediction']['modelData'] = modelData

        return jsonify(session['prediction'])
    else:
        return "This API accepts only POST requests"

@app.route("/getManualPrediction", methods=['POST', 'GET'])
def getManualPrediction():
    if request.method == "POST":
        global session

        fileName = request.form['fileName']
        openValue = request.form['openValue']
        highValue = request.form['highValue']
        lowValue = request.form['lowValue']
        volumeValue = request.form['volumeValue']

        prediction = getManualPredictionForModel(fileName, train_size, openValue, highValue, lowValue, volumeValue)

        session['prediction']['manualPrediction'] = str(prediction)
        return jsonify(session['prediction'])
    else:
        return "This API accepts only POST requests"