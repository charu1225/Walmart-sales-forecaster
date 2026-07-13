from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model file
with open('_model.pkl', 'rb') as file:
    model = pickle.load(file)

# API 1: Renders the initial web page
@app.route('/', methods=['GET'])
def home(): 
    return render_template('index.html', prediction=None)

# API 2: Handles the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    prediction_text = None

    try: 
        # Gathers the data typed into the form (matching HTML 'name' attributes)
        store = int(request.form['store'])
        holiday_flag = int(request.form['holiday_flag'])
        temperature = float(request.form['temperature'])
        fuel_price = float(request.form['fuel_price'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemployment'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])

        # Put those values in an array matching the model's expected order
        input_features = np.array([[store, holiday_flag, temperature, fuel_price, cpi, unemployment, day, month, year]])

        # Send the values to predict
        predicted_sales = model.predict(input_features)[0]

        # Format the prediction to 2 decimal places
        prediction_text = f"Weekly Predicted sales: ${predicted_sales:,.2f}"

    except Exception as e:
        prediction_text = f"Error: {str(e)}"

    # Return the template with the newly calculated prediction
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)