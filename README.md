FinSight Financial App
FinSight Fianacial App
FinSight is a full-stack financial web application designed to empower users with tools for analyzing, screening, and predicting stock market behavior. It combines modern UI design with backend intelligence using Flask, MongoDB, and deep learning models.

Features

User Authentication
Secure login and register functionality with session management.

Stock Market Analysis
Input stock symbols to retrieve historical performance and market trends.

Stock Screener
Filter and search stocks based on customized financial criteria.

Stock Price Prediction
LSTM-based model trained on Indian stock data for price forecasting.

Responsive UI
Designed with handcrafted frontend inspired by platforms like Groww and TradingView.

Smooth Scrolling & Section Navigation
One-page navigation with smooth user transitions across sections.


Tech Stack

Frontend: HTML, CSS (with Glassmorphism and Gradient Design), JavaScript

Backend: Python (Flask)

Database: SQLlite

Machine Learning: TensorFlow, yFinance, NumPy, Scikit-learn

Hosting: Render


Project Highlights

Modular codebase with separated templates and routes

Trained and saved LSTM models for efficient real-time predictions

Search auto-suggestions for symbols (planned enhancement)

Section-based dashboard with clear user flows and mobile responsiveness


Setup Instructions

1. Clone the repository:

git clone https://github.com/yourusername/FinSightFinancialApp.git
cd FinSightFinancialApp


2. Install dependencies:

pip install -r requirements.txt


3. Set up environment variables: Create a .env file and add your keys (MongoDB URI, API keys, etc.)


4. Run the application:

python app.py



Future Enhancements

Candlestick chart integration using Plotly or TradingView widget

More advanced screening filters

Real-time stock news and sentiment analysis

User portfolio saving and tracking

PWA (Progressive Web App) support


