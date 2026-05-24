<div align="center">
  <img src="https://img.shields.io/badge/FinSight-Financial_App-00ff88?style=for-the-badge&logo=flask&logoColor=black" alt="FinSight Logo">
  <h1>FinSight Financial App</h1>
  <p><b>Advanced tools for real-time stock analysis, LSTM-based predictions, and portfolio tracking.</b></p>

  <!-- Badges -->
  <a href="https://github.com/Vaseem546/FinSight-Financial-App/stargazers">
    <img src="https://img.shields.io/github/stars/Vaseem546/FinSight-Financial-App?style=flat-square&color=00ff88" alt="Stars" />
  </a>
  <a href="https://github.com/Vaseem546/FinSight-Financial-App/network/members">
    <img src="https://img.shields.io/github/forks/Vaseem546/FinSight-Financial-App?style=flat-square&color=00ff88" alt="Forks" />
  </a>
  <a href="https://github.com/Vaseem546/FinSight-Financial-App/issues">
    <img src="https://img.shields.io/github/issues/Vaseem546/FinSight-Financial-App?style=flat-square&color=00ff88" alt="Issues" />
  </a>
  <a href="https://render.com/">
    <img src="https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Deployed on Render" />
  </a>
</div>

<br />

FinSight is a full-stack financial web application providing retail investors with advanced tools for real-time stock analysis, Machine Learning (LSTM) predictions, and an elegant, responsive UI inspired by modern fintech platforms like TradingView and Groww.

---

## 🚀 Live Demo

**[👉 Click here to view the live application]([https://fin-sight-app.onrender.com](https://finsight-web-app.onrender.com))****

*(Add a screenshot or GIF of your dashboard here to make it eye-catching)*
<!-- <img src="link_to_your_screenshot.png" width="100%" alt="Dashboard Preview"> -->

---

## ✨ Features

* **User Authentication** — Secure login and registration with Flask and SQLite.
* **Interactive Stock Market Analysis** — Access 60 major stocks with dynamic, zoomable candlestick charts (powered by Plotly).
* **AI-Powered & Trend-Based Predictions** — Predict future stock prices using advanced LSTM neural networks. Includes a highly resilient **Graceful Fallback Mechanism** that automatically switches to algorithmic trend analysis if the server (like Render's free tier) lacks memory to load heavy ML models!
* **Advanced Stock Screener** — Filter and discover high-potential stocks based on a custom FinSight Score, RSI, and PE Ratio.
* **Portfolio Analytics** — Build, manage, and track portfolio risk, volatility, and diversification scores.
* **Responsive UI** — Mobile-first, dark-themed interface with smooth scroll and glassmorphism visual effects.
* **100% API-Independent** — All data is generated and synced locally using highly realistic market algorithms. No paid external API dependencies required!

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3, JavaScript, Plotly.js (for charts)
* **Backend:** Flask (Python 3.11), SQLAlchemy, Gunicorn
* **Database:** SQLite (API-independent, auto-seeding)
* **Data Science & ML:** TensorFlow / Keras (LSTM), Scikit-Learn, Pandas, NumPy
* **Deployment:** Render (Automated via `render.yaml`)

---

## 📂 Project Structure

```bash
FinSight-Financial-App/
├── app.py                        # Main Flask application & routing
├── backtesting.py                # ML prediction & backtesting logic
├── data_sync.py                  # Realistic market data generator
├── database.py                   # SQLAlchemy DB models & config
├── portfolio_analytics.py        # Portfolio risk & metrics algorithms
├── train_models.py               # LSTM neural network training script
├── setup.py                      # Database initialization & seeder
├── requirements.txt              # Python dependencies
├── render.yaml                   # Render deployment blueprint
├── static/
│   └── style.css                 # Global glassmorphism styles
├── templates/
│   ├── index.html                # Main dashboard & interactive UI
│   ├── portfolio.html            # Portfolio tracking page
│   └── login_register.html       # Auth pages
└── models/                       # Directory for pre-trained LSTM (.h5) models
```

---

## 💻 Getting Started (Local Development)

1. **Clone the repository:**
```bash
git clone https://github.com/Vaseem546/FinSight-Financial-App.git
cd FinSight-Financial-App
```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**
   * **PowerShell (Windows):** `.\venv\Scripts\Activate.ps1`
   * **Command Prompt (Windows):** `.\venv\Scripts\activate.bat`
   * **Mac/Linux:** `source venv/bin/activate`

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Initialize database and generate stock data:**
```bash
python setup.py
```

6. **Run the Flask server:**
```bash
python app.py
```

7. **Access the App:**
Open `http://127.0.0.1:5000` in your browser.

---

## ☁️ Deployment (Render)

This project is 100% pre-configured for deployment on **Render.com** using the included `render.yaml` Blueprint.

1. Connect your GitHub account to Render.
2. Create a new **Blueprint Instance**.
3. Select this repository.
4. Render will automatically run `pip install` and `python setup.py` to build the database, and launch the app using Gunicorn.

---

## 🧠 Optional: Train Models

To generate fresh Machine Learning predictions for the NIFTY 50 stocks, you can run the training script locally (Requires TensorFlow):
```bash
python train_models.py
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📜 License

MIT License © 2025 FinSight Team
