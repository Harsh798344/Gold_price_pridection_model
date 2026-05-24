from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
x_scaler = pickle.load(open("x_scaler.pkl", "rb"))
y_scaler = pickle.load(open("y_scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = [
            float(x.replace(',', ''))
            for x in request.form.values()
        ]

        final_features = np.array([features])

        # scale input
        scaled_features = x_scaler.transform(final_features)

        # predict
        prediction = model.predict(scaled_features)

        prediction = prediction.reshape(-1,1)

        # inverse transform
        real_price = y_scaler.inverse_transform(prediction)

        output = round(float(real_price[0][0]), 2)

        return render_template(
            'index.html',
            prediction_text=f'Predicted EUR/USD Rate: {output}$'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == "__main__":
    app.run(debug=True)