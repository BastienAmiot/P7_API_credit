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

logging.basicConfig(filename='app.log', level=logging.DEBUG)

model = pickle.load(open('lgbm_optimized.pkl', 'rb'))    

data =  ZipFile("data/main_test.zip")

@app.route('/', methods=['POST'])
def predict():
    try:
        input_data = request.json()
    
        user_id = input_data["user_id"]
        
        user_data = data[data['SK_ID_CURR'] == user_id]
    
        predictions = model.predict_proba(user_data[user_data.columns[1:]])
        predictions = predictions[:, 0]
    
        return jsonify(predictions.tolist())
    except Exception as e:
        app.logger.debug("Ceci est un message de débogage.")
        app.logger.error("Une erreur s'est produite : %s", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)))
