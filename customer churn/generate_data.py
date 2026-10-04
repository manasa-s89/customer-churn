"""
Synthetic OTT (Video Streaming) Customer Churn Dataset Generator
Generates realistic subscriber telemetry, viewing habits, billing friction, and retention signals.
"""

import os
import numpy as np
import pandas as pd


def generate_ott_churn_dataset(n_samples: int = 7500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OTT video streaming customer churn dataset.

    Features:
    - customer_id: Unique subscriber ID (e.g. "OTT-78421")
    - subscription_plan: 'Basic', 'Standard', 'Premium'
    - monthly_price: Monthly subscription fee ($7.99 to $22.99)
    - watch_hours_last_30_days: Total viewing hours in past 30 days (0 to 140 hrs)
    - days_since_last_watch: Days since subscriber last streamed content (0 to 45 days)
    - number_of_devices: Concurrent / registered devices (1 to 6)
    - number_of_profiles: Sub-profiles created on account (1 to 5)
    - downloads_count: Offline downloads in past 30 days (0 to 30)
    - customer_support_tickets: Technical or billing issues raised (0 to 6)
    - payment_failures: Number of payment declines/retries in past 90 days (0 to 3)
    - free_trial_converted: Whether subscriber originally converted from trial ('Yes' or 'No')
    - tenure_months: Duration of subscription in months (1 to 48)
    - churn_label: 1 if churned, 0 if active subscriber
    """
    rng = np.random.default_rng(random_state)

    # 1. Customer IDs
    id_nums = rng.integers(10000, 99999, size=n_samples)
    customer_ids = [f"OTT-{n}" for n in id_nums]

    # 2. Tenure in months (mix of new trial converts and long-term loyalists)
    mix = rng.random(n_samples)
    tenure_short = rng.integers(1, 7, size=n_samples)        # 1 - 6 months (high early risk)
    tenure_mid = rng.integers(7, 24, size=n_samples)         # 7 - 24 months
    tenure_long = rng.integers(24, 49, size=n_samples)       # 2 - 4 years
    tenure_months = np.where(mix < 0.40, tenure_short, np.where(mix < 0.75, tenure_mid, tenure_long))

    # 3. Subscription Plan
    # Basic (~45%), Standard (~35%), Premium (~20%)
    plan_choices = ['Basic', 'Standard', 'Premium']
    subscription_plan = rng.choice(plan_choices, size=n_samples, p=[0.45, 0.35, 0.20])

    # 4. Monthly Price (Plan pricing with occasional localized/legacy tiers or add-ons)
    monthly_price = np.zeros(n_samples)
    for i, plan in enumerate(subscription_plan):
        if plan == 'Basic':
            monthly_price[i] = rng.uniform(7.99, 9.99)
        elif plan == 'Standard':
            monthly_price[i] = rng.uniform(13.49, 15.99)
        else:
            monthly_price[i] = rng.uniform(19.99, 22.99)
    monthly_price = np.round(monthly_price, 2)

    # 5. Free Trial Converted ('Yes' / 'No')
    # Unconverted accounts or accounts with friction are less committed
    free_trial_converted = rng.choice(['Yes', 'No'], size=n_samples, p=[0.82, 0.18])

    # 6. Days Since Last Watch (0 to 45)
    # Loyal subscribers watch frequently (0-3 days); churning users become dormant (15-40 days)
    days_since_last_watch = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Correlate with tenure and trial status
        if free_trial_converted[i] == 'No' and tenure_months[i] <= 3:
            days_since_last_watch[i] = int(rng.exponential(scale=14))
        else:
            days_since_last_watch[i] = int(rng.exponential(scale=5))
    days_since_last_watch = np.clip(days_since_last_watch, 0, 45)

    # 7. Watch Hours in Last 30 Days (0.0 to 140.0 hours)
    # Inactive subscribers have very few hours, avid binge-watchers have 40+ hours
    watch_hours = np.zeros(n_samples)
    for i in range(n_samples):
        inactivity = days_since_last_watch[i]
        if inactivity >= 25:
            watch_hours[i] = rng.uniform(0.0, 3.5)
        elif inactivity >= 14:
            watch_hours[i] = rng.uniform(1.0, 12.0)
        elif inactivity >= 7:
            watch_hours[i] = rng.uniform(8.0, 28.0)
        else:
            watch_hours[i] = rng.uniform(22.0, 110.0)
    # Add random binge-watching bonus for Premium/Standard accounts
    binge_bonus = np.where(subscription_plan == 'Premium', rng.uniform(5.0, 30.0, size=n_samples), 0.0)
    watch_hours = np.round(np.clip(watch_hours + binge_bonus, 0.0, 140.0), 1)

    # 8. Number of Devices (1 to 6)
    # Premium accounts usually have 3-5 devices, Basic accounts 1-2
    devices = np.zeros(n_samples, dtype=int)
    for i, plan in enumerate(subscription_plan):
        if plan == 'Premium':
            devices[i] = rng.choice([2, 3, 4, 5, 6], p=[0.10, 0.25, 0.35, 0.20, 0.10])
        elif plan == 'Standard':
            devices[i] = rng.choice([1, 2, 3, 4], p=[0.20, 0.45, 0.25, 0.10])
        else:
            devices[i] = rng.choice([1, 2, 3], p=[0.60, 0.35, 0.05])
    number_of_devices = devices

    # 9. Number of Profiles (1 to 5)
    profiles = np.zeros(n_samples, dtype=int)
    for i, plan in enumerate(subscription_plan):
        if plan == 'Premium':
            profiles[i] = rng.choice([2, 3, 4, 5], p=[0.15, 0.35, 0.35, 0.15])
        elif plan == 'Standard':
            profiles[i] = rng.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.20, 0.10])
        else:
            profiles[i] = rng.choice([1, 2], p=[0.75, 0.25])
    number_of_profiles = profiles

    # 10. Downloads Count (Offline downloads in past 30 days)
    # Basic tier rarely uses downloads; active engaged users download mobile content
    downloads = rng.poisson(lam=np.where(watch_hours > 30, 6.0, 1.2))
    downloads_count = np.clip(downloads, 0, 30)

    # 11. Customer Support Tickets (0 to 6)
    # Buffering/playback issues, billing disputes
    ticket_lam = np.where(days_since_last_watch > 15, 0.5, 0.7)
    ticket_lam = np.where(monthly_price > 18.0, ticket_lam + 0.4, ticket_lam)
    tickets = rng.poisson(lam=ticket_lam)
    customer_support_tickets = np.clip(tickets, 0, 6)

    # 12. Payment Failures in past 90 days (0 to 3)
    # Involuntary churn driver: expired credit card or insufficient funds
    payment_fail_probs = [0.78, 0.15, 0.05, 0.02]
    payment_failures = rng.choice([0, 1, 2, 3], size=n_samples, p=payment_fail_probs)

    # 13. Realistic Churn Probability Model (Log-odds based on streaming psychology)
    # Base churn risk
    z = -1.25

    # Protective factors:
    z -= 0.045 * np.minimum(tenure_months, 36)                  # Long tenure stabilizes subscriber
    z -= 0.028 * np.minimum(watch_hours, 80.0)                 # Active streaming strongly protects
    z -= 0.18 * (number_of_profiles - 1)                       # Shared family profiles increase switching barrier
    z -= 0.08 * (number_of_devices - 1)                        # Multi-device ecosystem stickiness
    z -= 0.05 * np.minimum(downloads_count, 10)                # Offline downloads signal high reliance
    z -= 0.30 * (free_trial_converted == 'Yes')                # Explicit past conversion commitment

    # Risk drivers:
    z += 0.095 * days_since_last_watch                         # Inactivity / dormancy is #1 streaming churn driver
    z += 0.75 * payment_failures                               # Billing failures cause involuntary churn
    z += 0.45 * np.maximum(0, customer_support_tickets - 1)    # Recurring playback/billing complaints
    z += 0.35 * (subscription_plan == 'Basic')                 # Basic tier has lowest switching cost
    z += 0.035 * (monthly_price - 12.0)                        # Price sensitivity

    # Convert log-odds to probability
    churn_prob = 1.0 / (1.0 + np.exp(-z))
    churn_prob = np.clip(churn_prob, 0.01, 0.99)

    # Generate binary label with slight stochasticity
    churn_label = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "subscription_plan": subscription_plan,
        "monthly_price": monthly_price,
        "watch_hours_last_30_days": watch_hours,
        "days_since_last_watch": days_since_last_watch,
        "number_of_devices": number_of_devices,
        "number_of_profiles": number_of_profiles,
        "downloads_count": downloads_count,
        "customer_support_tickets": customer_support_tickets,
        "payment_failures": payment_failures,
        "free_trial_converted": free_trial_converted,
        "tenure_months": tenure_months,
        "churn_label": churn_label
    })

    return df


if __name__ == "__main__":
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ott_customer_churn.csv")
    df = generate_ott_churn_dataset(n_samples=7500, random_state=42)
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated {len(df)} synthetic OTT subscriber records -> {out_path}")
    print(f"     Churn Rate: {df['churn_label'].mean()*100:.1f}%")
    print(df.head())
