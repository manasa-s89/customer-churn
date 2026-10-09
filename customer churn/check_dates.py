import sqlite3
import pandas as pd

conn = sqlite3.connect("churn.db")
c = conn.cursor()
c.execute("SELECT min(created_at), max(created_at), count(*) FROM customers WHERE date(created_at) = '2026-10-09'")
print("2026-10-09 customers:", c.fetchall())

df = pd.read_sql("SELECT customer_id, tenure_months, subscription_plan, monthly_price, watch_hours_last_30_days, days_since_last_watch, customer_support_tickets, payment_failures FROM customers WHERE date(created_at) = '2026-10-09' LIMIT 10", conn)
print(df)
