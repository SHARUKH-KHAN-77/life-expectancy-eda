"""
generate_data.py
----------------
Generates a realistic synthetic Life Expectancy dataset modeled after the
WHO / World Bank Life Expectancy dataset (Kaggle: 'Life Expectancy (WHO)').

Run this script once to produce data/life_expectancy.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

COUNTRIES = {
    "Developed": [
        "United States", "Germany", "Japan", "United Kingdom", "France",
        "Canada", "Australia", "Sweden", "Norway", "Netherlands",
        "Switzerland", "Denmark", "Finland", "New Zealand", "Austria"
    ],
    "Developing": [
        "India", "Brazil", "China", "Nigeria", "Mexico",
        "Indonesia", "Pakistan", "Bangladesh", "Ethiopia", "Philippines",
        "Vietnam", "Kenya", "Ghana", "Tanzania", "Uganda",
        "Mozambique", "Zambia", "Zimbabwe", "Bolivia", "Cambodia"
    ],
}

YEARS = list(range(2000, 2016))

rows = []
for status, countries in COUNTRIES.items():
    for country in countries:
        base_le = np.random.uniform(74, 83) if status == "Developed" else np.random.uniform(52, 72)
        base_gdp = np.random.uniform(20000, 60000) if status == "Developed" else np.random.uniform(500, 8000)
        base_school = np.random.uniform(12, 17) if status == "Developed" else np.random.uniform(5, 12)

        for year in YEARS:
            trend = (year - 2000) * 0.15
            le = base_le + trend + np.random.normal(0, 0.4)
            gdp = base_gdp * (1 + 0.03 * (year - 2000)) + np.random.normal(0, base_gdp * 0.05)
            school = min(base_school + 0.1 * (year - 2000) + np.random.normal(0, 0.3), 20)
            adult_mort = max(np.random.normal(60 if status == "Developed" else 220, 20), 10)
            infant_d = max(np.random.normal(5 if status == "Developed" else 40, 5), 0)
            under5 = infant_d * np.random.uniform(1.1, 1.5)
            bmi = np.random.normal(27 if status == "Developed" else 23, 2)
            alcohol = np.random.normal(9 if status == "Developed" else 4, 2)
            hexp = np.random.normal(8 if status == "Developed" else 4, 1.5)
            pop = np.random.randint(1_000_000, 300_000_000)

            # Inject ~8% missing values in select columns
            for val_name in ["gdp", "school", "bmi", "alcohol", "hexp"]:
                if np.random.rand() < 0.08:
                    locals()[val_name] = np.nan

            rows.append({
                "Country": country,
                "Year": year,
                "Status": status,
                "Life expectancy": round(le, 1),
                "Adult Mortality": round(adult_mort, 0),
                "Infant deaths": round(infant_d, 0),
                "Under-five deaths": round(under5, 0),
                "GDP": round(gdp, 2),
                "Schooling": round(school, 1),
                "BMI": round(bmi, 1),
                "Alcohol": round(alcohol, 2),
                "percentage expenditure": round(hexp, 2),
                "Population": pop,
            })

df = pd.DataFrame(rows)

# Add a few duplicate rows intentionally
dups = df.sample(10, random_state=1)
df = pd.concat([df, dups], ignore_index=True)

df.to_csv("data/life_expectancy.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3))
