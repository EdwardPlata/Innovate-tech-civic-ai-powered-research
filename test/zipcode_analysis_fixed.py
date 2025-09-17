import pandas as pd

# Create a DataFrame
data = [
    {"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5},
    {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2},
    {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8},
    {"zipcode": "10004", "population": 8000, "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3},
    {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4},
    {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9},
    {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1},
    {"zipcode": "10008", "population": 19000, "median_income": 58000, "incident_count": 85, "area_sqmi": 0.6},
    {"zipcode": "10009", "population": 52000, "median_income": 45000, "incident_count": 280, "area_sqmi": 1.5},
    {"zipcode": "10010", "population": 38000, "median_income": 68000, "incident_count": 140, "area_sqmi": 1.0}
]

df = pd.DataFrame(data)

# Calculate incident rate per 1000 residents
df['incident_rate_per_1000'] = (df['incident_count'] / df['population']) * 1000

# Calculate incident density per square mile
df['incident_density_per_sqmi'] = (df['incident_count'] / df['area_sqmi'])

# Calculate percentiles
percent75 = df['incident_rate_per_1000'].quantile(0.75)
percent25 = df['incident_rate_per_1000'].quantile(0.25)

print("📊 ZIPCODE INCIDENT ANALYSIS")
print("=" * 60)

# Create risk level classification
def classify_risk(rate):
    if rate > percent75:
        return "High"
    elif rate < percent25:
        return "Low"
    else:
        return "Medium"

df['risk_level'] = df['incident_rate_per_1000'].apply(classify_risk)

# Create summary table
summary = df[['zipcode', 'population', 'incident_rate_per_1000', 'incident_density_per_sqmi', 'risk_level']].copy()

print("\n📈 FULL ANALYSIS RESULTS:")
print(summary.to_string(index=False, float_format='%.2f'))

# High-risk zipcodes
high_risk_codes = df[df['risk_level'] == 'High']['zipcode'].tolist()
low_risk_codes = df[df['risk_level'] == 'Low']['zipcode'].tolist()

print(f"\n🔥 HIGH-RISK ZIPCODES (above {percent75:.2f} incidents/1000): {', '.join(high_risk_codes)}")
print(f"🟢 LOW-RISK ZIPCODES (below {percent25:.2f} incidents/1000): {', '.join(low_risk_codes)}")

print(f"\n📊 STATISTICS:")
print(f"Average incident rate: {df['incident_rate_per_1000'].mean():.2f} per 1000 residents")
print(f"Highest incident rate: {df['incident_rate_per_1000'].max():.2f} (Zipcode {df.loc[df['incident_rate_per_1000'].idxmax(), 'zipcode']})")
print(f"Lowest incident rate: {df['incident_rate_per_1000'].min():.2f} (Zipcode {df.loc[df['incident_rate_per_1000'].idxmin(), 'zipcode']})")

print(f"\n🎯 RECOMMENDATIONS:")
for _, row in summary.iterrows():
    if row['risk_level'] == 'High':
        print(f"- Zipcode {row['zipcode']}: High incident rate ({row['incident_rate_per_1000']:.2f}/1000) - Consider increased patrols")
    elif row['risk_level'] == 'Low':
        print(f"- Zipcode {row['zipcode']}: Low incident rate ({row['incident_rate_per_1000']:.2f}/1000) - Good safety practices")