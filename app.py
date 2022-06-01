from flask import Flask, request
import joblib
import json
from cleanvec import Cleanvec
import numpy as np
import tensorflow as tf
app = Flask(__name__)


@app.route("/predecir", methods=['POST'])
def predict():
    if request.method == 'POST':
        cleanvec = Cleanvec()
        text = json.loads(request.data)['message']
        method = json.loads(request.data)['method']
        data, errors = ({}, [])

        try:
            if text == "":
                errors.append({"message": "message cannot be empty."})
            elif method == 'skl':
                model = joblib.load("sentiment_svm.pkl")
                clean_text = cleanvec.vectorized_text([text])
                my_prediction = int(model.predict(clean_text))
                prediction_map_sk = {0: 'Negative', 1: 'Positive', 2: 'Irrelevant'}
                data.update({
                    "text": text,
                    "prediction": prediction_map_sk.get(my_prediction)
                })
            elif method == 'dl':
                model = tf.keras.models.load_model('my_sentiment_model.h5')
                clean_text = cleanvec.padded_text([text])
                my_prediction = model.predict(clean_text)
                predicted_class = (np.argmax(my_prediction, axis=1)).item()
                prediction_map_dl = {0: 'Irrelevant', 1: 'Positive', 2: 'Negative'}
                data.update({
                    "text": text,
                    "prediction": prediction_map_dl.get(predicted_class)
                })
            else:
                errors.append({"message": "method must be either dl or skl"})

            data.update({"errors": errors})
            return data
        except Exception as e:
            return {"system error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)