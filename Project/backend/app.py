from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import glob
import pandas as pd
import os

from api import getRequiredColumns, LSTMAlgorithm, getPredictonsFromModel, getManualPredictionForModel

app = Flask("Stock Price Prediction")
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Scoped CORS to API routes

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

# Serve frontend static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.root_path, '../frontend', path)):
        return send_from_directory('../frontend', path)
    return send_from_directory('../frontend', 'index.html')

# API routes
@app.route('/api/', methods=['GET'])
def index():
    return "Welcome to Stock Price Prediction API"

@app.route('/api/upload', methods=['POST'])
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

@app.route('/api/startTraining', methods=['POST'])
def startTraining():
    if request.method == "POST":
        global session, df

        fileName = request.form['fileName']

        # Use Render disk mounts
        base_path = '/opt/render/project/src/datasets'
        os.makedirs(base_path, exist_ok=True)
        df.to_csv(os.path.join(base_path, f'{fileName}.csv'), index=False)

        session['training']['status'] = "training"
        session['training']['epochs'] = 0

        model = LSTMAlgorithm(fileName, train_size, totalEpochs, updateEpochs=updateEpochs)

        session['training']['status'] = "trainingCompleted"
        return jsonify(session['training'])
    else:
        return "This API accepts only POST requests"

@app.route('/api/trainingStatus', methods=['POST'])
def trainingStatus():
    if request.method == "POST":
        return jsonify(session['training'])
    else:
        return "This API accepts only POST requests"

@app.route('/api/getPreTrainedModels', methods=['POST'])
def getPreTrainedModels():
    if request.method == "POST":
        global session

        # Use Render disk mounts
        base_path = '/opt/render/project/src/pretrained'
        os.makedirs(base_path, exist_ok=True)
        files = glob.glob(os.path.join(base_path, '*.H5'))
        files = [f.split("/")[-1][:-3] for f in files]

        session['prediction']['preTrainedModelNames'] = files
        return jsonify(session['prediction'])
    else:
        return "This API accepts only POST requests"

@app.route('/api/getPredictions', methods=['POST'])
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

@app.route('/api/getManualPrediction', methods=['POST'])
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)  # Use for local testing with Render-compatible port