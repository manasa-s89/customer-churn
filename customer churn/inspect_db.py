import sqlite3

conn = sqlite3.connect("churn.db")
c = conn.cursor()

c.execute("SELECT count(*) FROM customers")
print("Total customers:", c.fetchone()[0], flush=True)

c.execute("SELECT count(*) FROM churn_scores")
print("Total churn scores:", c.fetchone()[0], flush=True)

c.execute("SELECT count(*) FROM interventions")
print("Total interventions:", c.fetchone()[0], flush=True)

c.execute("SELECT customer_id, tenure_months, monthly_price, subscription_plan, customer_support_tickets FROM customers LIMIT 5")
print("Sample customers:", c.fetchall(), flush=True)

c.execute("SELECT id, customer_id, action_type, status FROM interventions LIMIT 5")
print("Sample interventions:", c.fetchall(), flush=True)

conn.close()
