from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import glob
import pandas as pd
import os
import tempfile

from api import getRequiredColumns, LSTMAlgorithm, getPredictonsFromModel, getManualPredictionForModel

# Configure Flask app with custom static folder
app = Flask("Stock Price Prediction", static_folder='../frontend/static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

df = None
cols, dateColName, closeColName = None, None, None
train_size = 0.75
totalEpochs = 2

session = {
    "training": {
        "status": "ready",
        "fileUploaded": False,
        "fileName": None,
        "totalEpochs": totalEpochs,
        "tempFilePath": None
    },
    "prediction": {
        "status": "ready",
        "preTrainedModelNames": None
    }
}

def updateEpochs(epoch):
    global session
    session['training']['epochs'] = epoch + 1

# Serve frontend pages
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

        file = request.files['file']
        df = pd.read_csv(file)
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
        session['training']['fileName'] = file.filename[:-4]
        session['training']['cols'] = [dateColName] + cols
        session['training']['dfColVals'] = dfColVals
        session['training']['dfCloseVals'] = dfCloseVals
        session['training']['dfDateVals'] = dfDateVals
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
            session['training']['tempFilePath'] = temp_file.name
        
        return jsonify(session['training'])
    else:
        return "This API accepts only POST requests"

@app.route('/api/startTraining', methods=['POST'])
def startTraining():
    if request.method == "POST":
        global session, df

        fileName = request.form['fileName']
        tempFilePath = session['training']['tempFilePath']
        local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        local_pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')

        session['training']['status'] = "training"
        session['training']['epochs'] = 0

        try:
            # Load DataFrame from temporary file
            df = pd.read_csv(tempFilePath)
            model = LSTMAlgorithm(df, fileName, train_size, totalEpochs, updateEpochs=updateEpochs)
            os.makedirs(local_dataset_path, exist_ok=True)
            df.to_csv(os.path.join(local_dataset_path, f'{fileName}.csv'), index=False)
            os.makedirs(local_pretrained_path, exist_ok=True)
            model.save(os.path.join(local_pretrained_path, f'{fileName}.h5'))
            session['training']['status'] = "trainingCompleted"
            # Clean up temporary file
            os.unlink(tempFilePath)
            session['training']['tempFilePath'] = None
        except Exception as e:
            session['training']['status'] = "trainingFailed"
            print(f"Training failed: {str(e)}")
            if os.path.exists(tempFilePath):
                os.unlink(tempFilePath)
            return jsonify(session['training']), 500

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
        local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        print(f"Checking trained datasets in: {local_dataset_path}")
        os.makedirs(local_dataset_path, exist_ok=True)
        files = glob.glob(os.path.join(local_dataset_path, '*.csv'))
        files = [os.path.basename(f)[:-4] for f in files]  # Extract only the filename without .csv
        print(f"Found files: {files}")
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
    app.run(host='0.0.0.0', port=10000)