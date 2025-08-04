# FinSight Financial App

FinSight is a full-stack financial web application designed to deliver premium tools for retail investors. It combines real-time stock analysis, intelligent stock predictions, and a visually elegant UI inspired by modern fintech platforms like TradingView and Groww.

## Features

- **User Authentication**  
  Secure login and registration using Flask and MongoDB.

- **Stock Market Analysis**  
  Get live stock data, company overviews, and charts.

- **LSTM-Based Stock Price Prediction**  
  Predict future prices using pre-trained LSTM models for Nifty 50 stocks.

- **Stock Screener**  
  Filter and find high-potential stocks based on financial metrics.

- **Responsive UI Design**  
  Mobile-first and handcrafted with a premium aesthetic — smooth scroll, glassmorphism, and dark-themed UI.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask (Python)  
- **Database**: MongoDB  
- **ML Models**: TensorFlow, yFinance (for training)  
- **Deployment**: Render (or any preferred platform)

## Project Structure

FinSightFinancialApp/
│
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── login_register.html
├── models/
│   └── trained LSTM models
├── app.py
├── train_models.py
├── requirements.txt
└── README.md


## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FinSightFinancialApp.git
   cd FinSightFinancialApp

2.Install Dependencies 
```bash
pip install -r requirements.txt
