"""
FinSight Setup Script
Run this once to initialize the database and sync stock data
"""

print("🚀 FinSight Setup Starting...")

# Step 1: Initialize Database
print("\n📊 Step 1: Initializing database...")
from database import init_db
init_db()
print("✅ Database tables created")

# Step 2: Sync Stock Data
print("\n📈 Step 2: Syncing stock data (this may take 5-10 minutes)...")
from data_sync import sync_all_stocks
sync_all_stocks()

print("\n✅ Setup Complete!")
print("\n🎯 Next Steps:")
print("1. Run: python app_new.py")
print("2. Open: http://127.0.0.1:5000")
print("3. Register a new account")
print("4. Start analyzing stocks!")
print("\n💡 Tip: Run 'python data_sync.py' daily to update stock data")
