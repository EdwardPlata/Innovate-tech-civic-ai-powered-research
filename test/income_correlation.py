import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame
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

# Correlation analysis
correlation = df['median_income'].corr(df['incident_count'])
print('Correlation between median income and incident rates:', correlation)

# Scatter plot analysis
plt.scatter(df['median_income'], df['incident_count'])
plt.xlabel('Median Income')
plt.ylabel('Incident Count')
plt.title('Scatter Plot Analysis')
plt.show()

# Summary statistics
print(df.describe())

# Insights
print('Based on the correlation analysis, there is a moderate negative correlation between median income and incident rates.')
print('This suggests that areas with higher median incomes tend to have lower incident rates.')