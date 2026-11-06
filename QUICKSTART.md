# FinSight Quick Start

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Available Workflows

1. Register or log in.
2. Enter a stock symbol in Market Analysis.
3. Filter the locally generated stock universe in Stock Screener.
4. Add analyzed stocks to your portfolio.

Stock price prediction has been removed. Market history is fetched on demand from Yahoo Finance and cached in SQLite.

## Troubleshooting

- Re-run `python setup.py` if database tables are missing.
- Delete `finsight.db` only when you intentionally want to reset all users, portfolios, and demo data.
- Use one Flask worker during local development.
