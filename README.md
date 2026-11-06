# FinSight

FinSight is a small stock research dashboard built with Flask. It brings market data, technical indicators, stock screening, and personal portfolio tracking into one straightforward interface.

**Live demo:** https://fin-sight-financial-app.vercel.app

> Market data may be delayed. FinSight is an analysis tool and should not be used as financial advice or investment recommendations.

## What you can do

- Search stocks and view candlestick price charts
- Compare price, market capitalization, valuation, and 52-week highs and lows
- Explore indicators such as RSI and MACD
- Review a FinSight score calculated from available market metrics
- Screen stocks using score and technical indicator filters
- Create an account and keep a personal portfolio
- View portfolio-level value and performance summaries

## How it works

The app uses Flask for routes, templates, and sessions. SQLAlchemy manages the SQLite data model for users, portfolios, stock history, and calculated metrics. When a user looks up a stock, FinSight retrieves market history through `yfinance`, calculates the indicators, and renders an interactive Plotly chart.

The interface is built with standard HTML, CSS, and JavaScript, with no separate frontend build step.

## Technology

- Python 3 and Flask
- SQLAlchemy with SQLite
- pandas and NumPy
- Plotly
- Yahoo Finance data through `yfinance`
- HTML, CSS, and JavaScript
- Vercel for production hosting

## Run it locally

Requirements:

- Python 3.12 recommended
- Windows PowerShell, macOS terminal, or Linux shell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python setup.py
python app.py
```

Then open http://127.0.0.1:5000.

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Deploy to Vercel

The repository already includes [vercel.json](vercel.json). After installing and logging into the Vercel CLI:

```powershell
vercel login
vercel --prod
```

The production health check is available at:

```text
/health
```

## Database note

Locally, FinSight stores data in `finsight.db`. On Vercel, the app uses a SQLite database in `/tmp` because serverless functions do not provide persistent storage. Account and portfolio data may therefore change or reset between serverless deployments or cold starts.

For a persistent shared deployment, connect the app to an external database and configure it through an environment variable before deployment.

## Project structure

```text
app.py               Flask routes and application setup
database.py          SQLAlchemy models and database configuration
data_sync.py         Market data retrieval and technical calculations
portfolio_analytics.py Portfolio calculations
setup.py             Database initialization helper
templates/           Flask HTML templates
static/              Stylesheets and browser assets
vercel.json          Vercel deployment configuration
```

## Data notice

Stock history is fetched from Yahoo Finance through an unofficial public data interface. Availability, delays, symbols, and fields can change without notice. Historical and calculated values should be treated as informational rather than guaranteed.

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
