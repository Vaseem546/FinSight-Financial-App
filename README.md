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
```bash
FinSightFinancialApp/
├── app.py                        # Main Flask application
├── train_models.py              # LSTM model training script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys, DB URIs)
├── static/
│   ├── style.css                # Custom global styles
│   └── login_background.jpg     # Background image for auth pages
├── templates/
│   ├── index.html               # Main dashboard UI (after login)
│   ├── login_register.html     # Combined login/register interface
│   └── base.html                # Common layout (optional)
├── models/
│   ├── TCS_model.h5             # Example pre-trained LSTM model
│   └── ...                      # Other NIFTY50 stock models
├── utils/
│   └── helper_functions.py      # Data preprocessing, model loading, etc.
├── .gitignore                   # Ignore models, __pycache__, .env, etc.
└── README.md                    # Project overview and instructions


## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FinSightFinancialApp.git
   cd FinSightFinancialApp

2. Install dependencies:

pip install -r requirements.txt


3. Set up environment variables in a .env file:

OPENAI_API_KEY=your_openai_api_key


4. Run the Flask server:

python app.py


