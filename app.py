from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("capstone streamlit/rocket_model.pkl", "rb"))
columns = pickle.load(open("capstone streamlit/model_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    fuel_type = request.form['fuel_type']
    oxidizer_type = request.form['oxidizer_type']
    chamber_pressure = float(request.form['chamber_pressure'])
    oxidizer_fuel_ratio = float(request.form['oxidizer_fuel_ratio'])
    combustion_temperature = float(request.form['combustion_temperature'])
    heat_capacity_ratio = float(request.form['heat_capacity_ratio'])
    nozzle_expansion_ratio = float(request.form['nozzle_expansion_ratio'])
    ambient_pressure = float(request.form['ambient_pressure'])
    combustion_stability_margin = float(request.form['combustion_stability_margin'])

    input_dict = {
        "chamber_pressure_bar": chamber_pressure,
        "oxidizer_fuel_ratio": oxidizer_fuel_ratio,
        "combustion_temperature_K": combustion_temperature,
        "heat_capacity_ratio": heat_capacity_ratio,
        "nozzle_expansion_ratio": nozzle_expansion_ratio,
        "ambient_pressure_bar": ambient_pressure,
        "combustion_stability_margin": combustion_stability_margin,
        f"fuel_type_{fuel_type}": 1,
        f"oxidizer_type_{oxidizer_type}": 1
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=round(prediction,2))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)