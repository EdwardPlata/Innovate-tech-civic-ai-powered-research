import pandas as pd

# Create a DataFrame
data = [{"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5},
        {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2},
        {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8},
        {"zipcode": "10004", "population": 8000, "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3},
        {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4},
        {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9},
        {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1},
        {"zipcode": "10008", "population": 19000, "median_income": 58000, "incident_count": 85, "area_sqmi": 0.6},
        {"zipcode": "10009", "population": 52000, "median_income": 45000, "incident_count": 280, "area_sqmi": 1.5},
        {"zipcode": "10010", "population": 38000, "median_income": 68000, "incident_count": 140, "area_sqmi": 1.0}]

df = pd.DataFrame(data)

# Calculate incident rate per 1000 residents
df['incident_rate_per_1000'] = (df['incident_count'] / df['population']) * 1000

# Calculate incident density per square mile
df['incident_density_per_sqmi'] = (df['incident_count'] / df['area_sqmi'])

# Calculate percentiles
percent75 = df['incident_rate_per_1000'].quantile(0.75)
percent25 = df['incident_rate_per_1000'].quantile(0.25)

# Identify high-risk zipcodes (above 75th percentile)
high_risk = df[df['incident_rate_per_1000'] > percent75].copy()

# Identify low-risk zipcodes (below 25th percentile)
low_risk = df[df['incident_rate_per_1000'] < percent25].copy()

# Drop duplicates
high_risk = high_risk.drop_duplicates()
low_risk = low_risk.drop_duplicates()

# Create a summary table
summary = df[['zipcode', 'population', 'incident_rate_per_1000', 'incident_density_per_sqmi']]
summary['risk_level'] = None
for index, row in summary.iterrows():
    if row['incident_rate_per_1000'] > percent75:
        summary.loc[index, 'risk_level'] = "High"
    elif row['incident_rate_per_1000'] < percent25:
        summary.loc[index, 'risk_level'] = "Low"
    else:
        summary.loc[index, 'risk_level'] = "Medium"

# Get percentiles columns for final analysis result
high_risk_percentiles = []
low_risk_percentiles = []
for index, row in summary.iterrows():
    if row['risk_level'] == "High":
        high_risk_percentiles.append(row['zipcode'])
    elif row['risk_level