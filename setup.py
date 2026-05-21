"""
FinSight Setup Script
Run this once to initialize the database and sync stock data
"""

print("[SETUP] FinSight Setup Starting...")

# Step 1: Initialize Database
print("\n[STEP 1] Initializing database...")
from database import init_db
init_db()
print("[OK] Database tables created")

# Step 2: Sync Stock Data
print("\n[STEP 2] Generating stock data locally...")
from data_sync import sync_all_stocks
sync_all_stocks()

print("\n[DONE] Setup Complete!")
print("\n[INFO] Next Steps:")
print("1. Run: python app_new.py")
print("2. Open: http://127.0.0.1:5000")
print("3. Register a new account")
print("4. Start analyzing stocks!")
