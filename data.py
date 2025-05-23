import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_inquiry_data(n_samples=2000):
    """Generate synthetic inquiry data with realistic web server log attributes and hourly distribution."""
    np.random.seed(42)
    random.seed(42)

    # Define static lists
    salespersons = ["Momphitisi", "Lerato", "Tshepo", "Kagiso", "Naledi"]
    products = ["AI-Chatbot", "AI-Analytics", "AI-Vision", "AI-Automation"]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders = ["Male", "Female", "Other"]
    countries = ["South Africa", "Nigeria", "Kenya", "USA", "UK", "India", "Brazil", "Germany", "Japan", "Australia"]
    continents = ["Africa", "Africa", "Africa", "North America", "Europe", "Asia", "South America", "Europe", "Asia", "Oceania"]
    country_to_continent = dict(zip(countries, continents))
    referral_sources = ["Social Media", "Email Campaign", "Website", "Partner", "Direct"]

    # Define web server log patterns
    http_methods = ["GET", "POST", "PUT"]
    urls = ["/api/inquiries", "/api/submit", "/api/update", "/api/details", "/api/error"]
    status_codes = [200, 201, 400, 404, 500]
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "curl/7.68.0",
        "Python-urllib/3.9"
    ]

    # Generate daily timestamps with random hours
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 5, 19)
    date_range = (end_date - start_date).days
    timestamps = [start_date + timedelta(days=i) for i in range(date_range + 1)]
    n_samples_per_day = n_samples // len(timestamps) + 1
    timestamps = [t for t in timestamps for _ in range(n_samples_per_day)][:n_samples]
    
    # Add random hours to timestamps
    hours = [random.randint(0, 23) for _ in range(n_samples)]
    timestamps = [ts + timedelta(hours=h) for ts, h in zip(timestamps, hours)]

    # Generate core data
    data = {
        "Timestamp": timestamps,
        "Inquiries": np.random.poisson(lam=5, size=n_samples).clip(min=0),
        "PreviousInquiries": np.random.poisson(lam=4, size=n_samples).clip(min=0),
        "Salesperson": [random.choice(salespersons) for _ in range(n_samples)],
        "Product": [random.choice(products) for _ in range(n_samples)],
        "AgeGroup": [random.choice(age_groups) for _ in range(n_samples)],
        "Gender": [random.choice(genders) for _ in range(n_samples)],
        "Country": [random.choice(countries) for _ in range(n_samples)],
        "ReferralSource": [random.choice(referral_sources) for _ in range(n_samples)],
    }

    # Map Continent based on Country
    data["Continent"] = [country_to_continent[country] for country in data["Country"]]

    # Extract Hour from Timestamp
    data["Hour"] = [ts.hour for ts in data["Timestamp"]]

    # Generate TargetInquiries with hourly variation (higher during business hours 9-17)
    hour_weights = [1.5 if 9 <= h <= 17 else 0.5 for h in data["Hour"]]
    data["TargetInquiries"] = [max(0, int(np.random.poisson(lam=6) * w)) for w in hour_weights]

    # Generate Sales based on Inquiries to ensure positive correlation
    inquiries = np.array(data["Inquiries"])
    base_sales = 5 + 3 * inquiries + np.random.normal(0, 2, n_samples)  # Sales = 5 + 3*Inquiries + noise
    data["Sales"] = np.clip(base_sales, 0, None).astype(int)

    # Generate ForecastedSales with a slight offset
    data["ForecastedSales"] = (data["Sales"] * np.random.uniform(0.95, 1.05, n_samples)).round().astype(int)

    # Generate realistic web server log attributes
    web_logs = []
    for _ in range(n_samples):
        scenario = random.choice(["Browse", "Submit", "Update", "Error"])
        
        if scenario == "Browse":
            method = "GET"
            url = "/api/inquiries"
            status = random.choices([200, 404], weights=[0.9, 0.1])[0]
            request_size = random.randint(300, 800)
            user_agent = random.choice(user_agents[:3])
            ip = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"
        
        elif scenario == "Submit":
            method = "POST"
            url = "/api/submit"
            status = random.choices([201, 400], weights=[0.8, 0.2])[0]
            request_size = random.randint(1500, 3000)
            user_agent = random.choice(user_agents[:3])
            ip = f"10.0.{random.randint(0, 255)}.{random.randint(0, 255)}"
        
        elif scenario == "Update":
            method = "PUT"
            url = "/api/update"
            status = random.choices([200, 400], weights=[0.85, 0.15])[0]
            request_size = random.randint(1000, 2000)
            user_agent = random.choice(user_agents[:3])
            ip = f"172.16.{random.randint(0, 255)}.{random.randint(0, 255)}"
        
        else:  # Error scenario
            method = random.choice(["GET", "POST"])
            url = "/api/error"
            status = random.choices([500, 404], weights=[0.7, 0.3])[0]
            request_size = random.randint(500, 1500)
            user_agent = random.choice(user_agents)
            ip = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"

        web_logs.append({
            "HttpMethod": method,
            "StatusCode": status,
            "Url": url,
            "UserAgent": user_agent,
            "IPAddress": ip,
            "RequestSize": request_size
        })

    # Add web server log attributes to the dataset
    web_log_df = pd.DataFrame(web_logs)
    for col in web_log_df.columns:
        data[col] = web_log_df[col]

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = generate_inquiry_data()
    df.to_csv("inquiry_data.csv", index=False)
    print("Generated inquiry_data.csv with realistic web server logs, daily timestamps, hourly distribution, and positive Sales-Inquiries correlation.")