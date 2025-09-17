import pandas as pd

# Create DataFrame from the provided data
data = [
    {"zipcode": "10001", "population": 23000, "median_income": 65000, "incident_count": 120, "area_sqmi": 0.5, "police_stations": 2},
    {"zipcode": "10002", "population": 42000, "median_income": 48000, "incident_count": 210, "area_sqmi": 1.2, "police_stations": 3},
    {"zipcode": "10003", "population": 31000, "median_income": 72000, "incident_count": 95, "area_sqmi": 0.8, "police_stations": 2},
    {"zipcode": "10004", "population": 8000, "median_income": 90000, "incident_count": 15, "area_sqmi": 0.3, "police_stations": 1},
    {"zipcode": "10005", "population": 15000, "median_income": 85000, "incident_count": 45, "area_sqmi": 0.4, "police_stations": 1},
    {"zipcode": "10006", "population": 28000, "median_income": 55000, "incident_count": 160, "area_sqmi": 0.9, "police_stations": 2},
    {"zipcode": "10007", "population": 35000, "median_income": 62000, "incident_count": 180, "area_sqmi": 1.1, "police_stations": 3},
    {"zipcode": "10008", "population": 19000, "median_income": 58000, "incident_count": 85, "area_sqmi": 0.6, "police_stations": 2},
    {"zipcode": "10009", "population": 52000, "median_income": 45000, "incident_count": 280, "area_sqmi": 1.5, "police_stations": 4},
    {"zipcode": "10010", "population": 38000, "median_income": 68000, "incident_count": 140, "area_sqmi": 1.0, "police_stations": 3}
]
df = pd.DataFrame(data)

# Analyze population density vs incident density
df["population_density"] = df["population"] / (df["area_sqmi"] * 1e6)  # population per square mile
df["incident_density"] = df["incident_count"] / (df["area_sqmi"] * 1e6)  # incidents per square mile

# Perform the requested analysis
print("Summary statistics:")
print(df.describe())

# Find the correlation between population density and incident density
corr = df[["population_density", "incident_density"]].corr()
print("\nCorrelation between population density and incident density:")
print(corr)

# Find the area with the highest incident density
highest_incident_density_area = df.loc[df["incident_density"].idxmax()]
print(f"\nThe area with the highest incident density is {highest_incident_density_area['zipcode']} with {highest_incident_density_area['incident_count']} incidents per square mile.")

# Find the area with the lowest police station to population ratio
lowest_police_station_ratio_area = df.loc[df["area_sqmi"] / df["police_stations"].corr()
                                         .astype(float)].sort_values(by="area_sqmi / police_stations").head(1)
print(f"\nThe area with the lowest police station to population ratio is {lowest_police_station_ratio_area['zipcode']} with {lowest_police_station_ratio_area['area_sqmi / police_stations']}.")