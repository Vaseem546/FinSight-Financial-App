
## FinSight Financial App

FinSight is a full-stack financial web application that empowers retail investors with advanced tools for **real-time stock analysis**, **LSTM-based predictions**, and a sleek, responsive UI inspired by modern fintech platforms like TradingView and Groww.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Database Configuration](#database-configuration)
- [Usage](#usage)
- [Optional: Train Models](#optional-train-models)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **User Authentication** — Secure login and registration with Flask and MySQL.  
- **Stock Market Analysis** — Access live stock data, company profiles, and interactive charts.  
- **LSTM-Based Stock Price Prediction** — Predict future stock prices for Nifty 50 using pre-trained models.  
- **Stock Screener** — Filter and discover high-potential stocks based on key metrics.  
- **Responsive UI** — Mobile-first, dark-themed interface with smooth scroll and glassmorphism effects.

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python 3.11)  
- **Database:** MySQL  
- **ML Models:** TensorFlow, yFinance  
- **Deployment:** Render

---

## Project Structure

```bash
FinSightFinancialApp/
├── app.py                        # Main Flask application
├── train_models.py               # LSTM model training script
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API keys, DB credentials)
├── static/
│   ├── style.css                 # Global styles
│   └── login_background.jpg      # Auth page background
├── templates/
│   ├── index.html                # Dashboard after login
│   ├── login_register.html       # Combined login/register page
│   └── base.html                 # Common layout (optional)
├── models/
│   ├── TCS_model.h5              # Pre-trained LSTM model
│   └── ...                       # Other NIFTY50 stock models
├── utils/
│   └── helper_functions.py       # Data preprocessing and model helpers
├── .gitignore                    # Ignore models, __pycache__, .env, etc.
└── README.md                     # Project documentation
````

> **Note:** `train_models.py` allows you to train or update the LSTM models for Nifty 50 stocks.

---

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/FinSightFinancialApp.git
cd FinSightFinancialApp
```

2. **Create and activate a virtual environment:**

*Windows PowerShell:*

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*Command Prompt:*

```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables in a `.env` file:**

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=finsight_db
```

> ⚠️ Ensure `.env` is **not committed** to GitHub.

---

## Database Configuration

FinSight uses **MySQL** as its database. Credentials are loaded from the `.env` file for security.

### Connecting in the App

```python
from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
```

> Ensure the MySQL database (`DB_NAME`) exists before running the app.

---

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open your browser at:

```
http://127.0.0.1:5000
```

3. Register or login, then access real-time stock data, predictions, and the stock screener.

---

## Optional: Train Models

```bash
python train_models.py
```

Use this script to train or update LSTM models for Nifty 50 stocks.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push the branch: `git push origin feature-name`
5. Open a Pull Request

---

## License

MIT License © 2025 FinSight Team

```

