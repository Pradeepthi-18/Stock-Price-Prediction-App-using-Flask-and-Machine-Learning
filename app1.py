from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from alpha_vantage.timeseries import TimeSeries
import os
import time
import logging
import joblib

# Initialize Flask app
app = Flask(__name__)

# Logging Configuration
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Alpha Vantage API Key (stored as an environment variable)
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

if not API_KEY:
    raise ValueError("API Key for Alpha Vantage is not set.")

# Fetch stock data using Alpha Vantage
def fetch_stock_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    data = data.sort_index(ascending=True)  # Sort by date
    return data

# Train ML model for prediction
def train_model(data):
    data['Date'] = np.arange(len(data))  # Convert dates into numeric format
    X = data[['Date']]
    y = data['4. close']  # Target is the closing price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection: Using Random Forest Regressor for better accuracy
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Cache the model to avoid retraining every time
def save_model(model):
    joblib.dump(model, 'stock_price_model.pkl')

def load_model():
    return joblib.load('stock_price_model.pkl')

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle stock prediction
@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']

    try:
        data = fetch_stock_data(symbol)
        
        # Check if model exists, if not, train the model
        if os.path.exists('stock_price_model.pkl'):
            model = load_model()
            logging.info(f"Loaded model for prediction.")
        else:
            model, X_test, y_test = train_model(data)
            save_model(model)
            logging.info(f"Trained and saved new model for {symbol}.")
        
        # Predict next day's stock price
        next_day = [[len(data)]]  # Next day's numeric representation
        predicted_price = model.predict(next_day)[0]

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save plot with unique filename
        plot_filename = f'static/plot_{time.time()}.png'
        plt.figure(figsize=(10, 5))
        plt.plot(data['4. close'], label='Closing Prices', color='blue')
        plt.title(f'Stock Prices for {symbol}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(plot_filename)
        plt.close()

        return render_template('result.html',
                               symbol=symbol,
                               predicted_price=round(predicted_price, 2),
                               mse=round(mse, 2),
                               r2=round(r2, 2),
                               plot_path=plot_filename)
    except Exception as e:
        logging.error(f"Error in prediction for {symbol}: {e}")
        user_friendly_message = "There was an issue fetching data or processing your request. Please try again."
        return render_template('error.html', error=user_friendly_message)

if __name__ == '__main__':
    app.run(debug=True)
