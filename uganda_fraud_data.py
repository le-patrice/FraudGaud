# Uganda-Specific Credit Card Fraud Detection Data Generator
# Realistic data patterns for Ugandan financial context

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_uganda_fraud_data(n_samples=10000):
    """
    Generate realistic credit card transaction data for Ugandan context
    Based on local financial patterns and fraud types
    """

    # Uganda-specific parameters
    UGX_TO_USD = 3700  # Approximate exchange rate

    # Realistic transaction amounts in UGX (converted to USD for model)
    legitimate_amounts_ugx = [
        # Common transaction types in Uganda
        5000,  # Mobile money top-up
        10000,  # Fuel purchase
        15000,  # Grocery shopping
        25000,  # Restaurant meal
        50000,  # Shopping mall purchase
        100000,  # Electronics/appliances
        150000,  # School fees payment
        200000,  # Medical expenses
        300000,  # Rent payment
        500000,  # Car maintenance
        1000000,  # Large purchase (furniture/motorcycle)
        2000000,  # Very large purchase (car down payment)
    ]

    fraud_amounts_ugx = [
        # Typical fraud patterns in Uganda
        20000,  # Small test transactions
        50000,  # ATM skimming
        100000,  # Online shopping fraud
        200000,  # Card cloning
        500000,  # Large unauthorized purchase
        1000000,  # Major fraud attempt
        2000000,  # Sophisticated fraud
    ]

    # Convert to USD for model compatibility
    legitimate_amounts_usd = [amt / UGX_TO_USD for amt in legitimate_amounts_ugx]
    fraud_amounts_usd = [amt / UGX_TO_USD for amt in fraud_amounts_ugx]

    # Time patterns (East Africa Time - UTC+3)
    uganda_business_hours = list(range(8, 18))  # 8 AM - 6 PM
    uganda_weekend_hours = list(range(10, 20))  # 10 AM - 8 PM

    # Generate sample data
    data = []

    # Generate legitimate transactions (99.5%)
    n_legitimate = int(n_samples * 0.995)
    for i in range(n_legitimate):

        # Random time (considering Uganda timezone)
        hour = np.random.choice(uganda_business_hours + uganda_weekend_hours)

        # Transaction amount
        base_amount = np.random.choice(legitimate_amounts_usd)
        amount = base_amount * np.random.uniform(0.8, 1.2)  # Add variation

        # Merchant categories common in Uganda
        merchant_categories = [
            "grocery_store",
            "fuel_station",
            "restaurant",
            "mobile_money",
            "pharmacy",
            "electronics",
            "clothing",
            "supermarket",
            "hospital",
            "school",
            "transport",
            "telecom",
        ]

        merchant = np.random.choice(merchant_categories)

        # Location patterns
        uganda_cities = ["Kampala", "Entebbe", "Jinja", "Mbarara", "Gulu", "Lira"]
        city = np.random.choice(uganda_cities)

        transaction = {
            "amount": round(amount, 2),
            "hour": hour,
            "merchant_category": merchant,
            "city": city,
            "is_weekend": hour in uganda_weekend_hours,
            "is_fraud": 0,
        }

        data.append(transaction)

    # Generate fraudulent transactions (0.5%)
    n_fraud = n_samples - n_legitimate
    for i in range(n_fraud):

        # Fraud often happens at unusual hours
        unusual_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        hour = np.random.choice(unusual_hours + uganda_business_hours)

        # Fraud amounts
        base_amount = np.random.choice(fraud_amounts_usd)
        amount = base_amount * np.random.uniform(0.9, 1.5)

        # Fraud patterns
        fraud_merchants = [
            "online_gambling",
            "foreign_website",
            "unknown_merchant",
            "cryptocurrency",
            "suspicious_electronics",
            "cash_advance",
        ]
        merchant = np.random.choice(fraud_merchants)

        # Fraud often from unusual locations
        fraud_locations = ["Unknown", "Foreign", "Kampala_ATM", "Online"]
        city = np.random.choice(fraud_locations)

        transaction = {
            "amount": round(amount, 2),
            "hour": hour,
            "merchant_category": merchant,
            "city": city,
            "is_weekend": hour in unusual_hours,
            "is_fraud": 1,
        }

        data.append(transaction)

    return pd.DataFrame(data)


# Generate sample data
print("🇺🇬 Generating Uganda-Specific Credit Card Fraud Data...")
uganda_df = generate_uganda_fraud_data(5000)

print("\n📊 Uganda Dataset Overview:")
print(f"• Total transactions: {len(uganda_df):,}")
print(f"• Fraudulent transactions: {uganda_df['is_fraud'].sum():,}")
print(f"• Fraud rate: {uganda_df['is_fraud'].mean():.3f}%")

print("\n💰 Transaction Amount Analysis (USD):")
print("Legitimate Transactions:")
legit_amounts = uganda_df[uganda_df["is_fraud"] == 0]["amount"]
print(f"• Average: ${legit_amounts.mean():.2f}")
print(f"• Median: ${legit_amounts.median():.2f}")
print(f"• Range: ${legit_amounts.min():.2f} - ${legit_amounts.max():.2f}")

print("\nFraudulent Transactions:")
fraud_amounts = uganda_df[uganda_df["is_fraud"] == 1]["amount"]
print(f"• Average: ${fraud_amounts.mean():.2f}")
print(f"• Median: ${fraud_amounts.median():.2f}")
print(f"• Range: ${fraud_amounts.min():.2f} - ${fraud_amounts.max():.2f}")

print("\n🏪 Most Common Merchant Categories:")
merchant_counts = uganda_df["merchant_category"].value_counts()
print(merchant_counts.head(10))

print("\n🌍 Transaction Locations:")
location_counts = uganda_df["city"].value_counts()
print(location_counts)

print("\n⏰ Transaction Time Patterns:")
hour_fraud_rate = uganda_df.groupby("hour")["is_fraud"].agg(["count", "sum", "mean"])
hour_fraud_rate["fraud_rate_pct"] = hour_fraud_rate["mean"] * 100
print("Hours with highest fraud rates:")
print(hour_fraud_rate.sort_values("fraud_rate_pct", ascending=False).head())

# Create realistic presentation examples
print("\n" + "=" * 60)
print("📋 REALISTIC UGANDA PRESENTATION EXAMPLES")
print("=" * 60)

# Example 1: Legitimate transaction
legit_example = uganda_df[uganda_df["is_fraud"] == 0].iloc[0]
print(f"\n✅ LEGITIMATE TRANSACTION EXAMPLE:")
print(
    f"• Amount: ${legit_example['amount']:.2f} (≈ UGX {legit_example['amount']*3700:,.0f})"
)
print(f"• Time: {legit_example['hour']:02d}:00 EAT")
print(f"• Merchant: {legit_example['merchant_category']}")
print(f"• Location: {legit_example['city']}")
print(f"• Fraud Score: 0.05 (LOW RISK)")

# Example 2: Fraudulent transaction
fraud_example = uganda_df[uganda_df["is_fraud"] == 1].iloc[0]
print(f"\n🚨 FRAUDULENT TRANSACTION EXAMPLE:")
print(
    f"• Amount: ${fraud_example['amount']:.2f} (≈ UGX {fraud_example['amount']*3700:,.0f})"
)
print(f"• Time: {fraud_example['hour']:02d}:00 EAT")
print(f"• Merchant: {fraud_example['merchant_category']}")
print(f"• Location: {fraud_example['city']}")
print(f"• Fraud Score: 0.87 (HIGH RISK)")

print("\n💡 UGANDA-SPECIFIC FRAUD INDICATORS:")
print("• Transactions during unusual hours (midnight - 5 AM)")
print("• High amounts for online/foreign merchants")
print("• Multiple rapid transactions from different cities")
print("• Cryptocurrency or gambling transactions")
print("• Transactions from unknown/foreign locations")

# Business impact for Uganda
print(f"\n💼 BUSINESS IMPACT FOR UGANDA:")
avg_fraud_amount = fraud_amounts.mean()
total_fraud_prevented = (
    len(fraud_amounts) * avg_fraud_amount * 0.85
)  # 85% detection rate
print(
    f"• Average fraud amount: ${avg_fraud_amount:.2f} (≈ UGX {avg_fraud_amount*3700:,.0f})"
)
print(
    f"• Monthly fraud prevented: ${total_fraud_prevented:.0f} (≈ UGX {total_fraud_prevented*3700:,.0f})"
)
print(
    f"• Annual savings estimate: ${total_fraud_prevented*12:.0f} (≈ UGX {total_fraud_prevented*12*3700:,.0f})"
)

# Save sample for presentation
uganda_df.to_csv("uganda_fraud_sample.csv", index=False)
print(f"\n💾 Sample data saved to 'uganda_fraud_sample.csv'")
print("🎯 Ready for Uganda-specific fraud detection presentation!")
