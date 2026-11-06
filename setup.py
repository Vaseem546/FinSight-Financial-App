"""Initialize the FinSight SQLite database."""

print("[SETUP] FinSight Setup Starting...")

from database import init_db
init_db()

print("[DONE] Setup Complete!")
print("Run: python app.py")
print("Open: http://127.0.0.1:5000")
