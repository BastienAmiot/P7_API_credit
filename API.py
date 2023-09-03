import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from zipfile import ZipFile
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)

model = pickle.load(open('lgbm_optimized.pkl', 'rb'))    

data =  ZipFile("data/main_test.zip")

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Hello World</h1>'''

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json()
    logging.debug("Requête JSON reçue : %s", input_data)
    
    user_id = input_data["user_id"]
    logging.debug("Identifiant utilisateur : %s", user_id)
    
    user_data = data[data['SK_ID_CURR'] == user_id]
    logging.debug("Données utilisateur: %s", user_data)
    
    predictions = model.predict_proba(user_data[user_data.columns[1:]])
    logging.debug("Probabilités de prédiction: %s", predictions)
    predictions = predictions[:, 0]
    logging.debug("Probabilité de solvabilité de l'utilisateur: %s", predictions)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)))
