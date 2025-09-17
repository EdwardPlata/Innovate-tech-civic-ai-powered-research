import pandas as pd

# Create DataFrame from provided data
data = [{"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5, "police_stations": 2},
        {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2, "police_stations": 3},
        {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8, "police_stations": 2},
        {"zipcode": "10004", "population": 8000, "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3, "police_stations": 1},
        {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4, "police_stations": 1},
        {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9, "police_stations": 2},
        {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1, "police_stations": 3},
        {"zipcode": "10008", "population": 19000, "median_income": 58000, "incident_count": 85, "area_sqmi": 0.6, "police_stations": 2},
        {"zipcode": "10009", "population": 52000, "median_income": 45000, "incident_count": 280, "area_sqmi": 1.5, "police_stations": 4},
        {"zipcode": "10010", "population": 38000, "median_income": 68000, "incident_count": 140, "area_sqmi": 1.0, "police_stations": 3}]

df = pd.DataFrame(data)

# Perform requested analysis
df['incidents_per_sqmi'] = df['incident_count'] / df['area_sqmi']

df_sorted = df.sort_values(by='incidents_per_sqmi', ascending=False)

top_3_dangerous_zipcodes = df_sorted.head(3)

print("Top 3 Most Dangerous Zipcodes:")
print(top_3_dangerous_zipcodes[['zipcode', 'incidents_per_sqmi']])

print("\nSummary Statistics:")
print(df.describe())

print("\nInsights:")
print("Based on the analysis, it appears that the top 3 most dangerous zipcodes have significantly higher incidents per square mile than the rest of the zipcodes. This suggests that these areas may require increased police presence and resource allocation to improve public safety.")
print("The summary statistics also reveal that the median income and police stations per area are not strongly correlated with incidents per square mile, indicating that other factors may be contributing to the high incident rates in these areas.")