# Stock Price Predictor App

**Overview :**

This project is a simple Stock Price Predictor web application built using Streamlit. It allows users to input a stock ticker (e.g., TCS.NS), fetches historical stock data from Yahoo Finance, and predicts future stock prices using a pre-trained Neural Network model. The application also displays various Moving Averages (MA) for different periods (100, 200, 250 days) along with a comparison between the original and predicted stock prices.

**Features :**

1. **User Input for Stock Ticker :** Users can input any stock symbol to fetch historical data.
2. **Data Visualization :** Displays original stock prices along with moving averages for 100, 200, and 250 days.
3. **Stock Prediction :** Predicts stock prices based on a pre-trained model and compares them to actual stock prices.
4. **Plot Comparison :** Visualization comparing predicted and original stock prices for easy analysis.

**Tools and Libraries Used :**
1. Python
2. Streamlit (for building the interactive web app)
3. TensorFlow/Keras (for Neural Network model)
4. Pandas (for data manipulation and analysis)
5. NumPy (for numerical computations)
6. Matplotlib/Seaborn (for data visualization)
7. yFinance (for downloading historical stock data)
8. Scikit-learn (for data scaling MinMaxScaler)

**Project Structure :**
1. **data/:** Contains historical stock price data.
2. **models/:** Stores trained models.
3. **notebooks/:** Jupyter notebooks for experimentation.
4. **src/:** Source code for data processing, model building, and prediction.

**How the Prediction Works :**
1. **Data Preprocessing :** The stock's closing prices are scaled using a MinMaxScaler to normalize the data.
2. **Model Prediction :** The pre-trained neural network model predicts future prices based on past 100 days of stock price data.
3. **Inverse Transformation :** Predicted values are scaled back to the original price scale for comparison with the actual stock prices.
