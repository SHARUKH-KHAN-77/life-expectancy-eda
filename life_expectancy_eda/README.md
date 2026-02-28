# 🌍 Global Life Expectancy & Health Data Analysis

> **An industry-quality EDA project** analysing WHO/World Bank-style life expectancy data  
> using Python · Pandas · Seaborn · Matplotlib · Scikit-learn

---

## 📋 Project Overview

This project investigates **what drives life expectancy** across 35 countries from 2000 to 2015.  
Through rigorous exploratory data analysis (EDA), feature engineering, and a baseline ML model,  
we uncover the interplay of health, economic, and social factors.

Key questions answered:
- Which features correlate most strongly with life expectancy?
- How has global life expectancy trended over 15 years?
- Can we predict life expectancy with simple linear regression?
- How do Developed vs Developing nations differ across all health metrics?

---

## 🗂️ Project Structure

```
life_expectancy_eda/
│
├── data/
│   ├── generate_data.py        ← Generates synthetic dataset (run once)
│   ├── life_expectancy.csv     ← Raw dataset
│   └── life_expectancy_clean.csv  ← Cleaned dataset
│
├── notebooks/
│   └── eda_analysis.ipynb      ← Full EDA walkthrough with markdown
│
├── reports/
│   ├── eda_profile.html        ← Standalone HTML report (open in browser)
│   ├── 01_life_expectancy_distribution.png
│   ├── 02_trend_over_years.png
│   ├── 03_gdp_vs_life_expectancy.png
│   ├── 04_schooling_vs_life_expectancy.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_status_comparison.png
│   ├── 07_pairplot.png
│   ├── 08_top_bottom_countries.png
│   └── 09_feature_coefficients.png
│
├── src/
│   ├── data_cleaning.py        ← Dedup, impute, outlier treatment
│   ├── visualization.py        ← All Matplotlib/Seaborn plots
│   └── feature_engineering.py ← Scaling, encoding, derived features
│
├── run_eda.py                  ← 🚀 Master script — runs everything
├── build_notebook.py           ← Generates the Jupyter notebook
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

| Column | Description |
|---|---|
| `Country` | Country name |
| `Year` | Year (2000–2015) |
| `Status` | Developed / Developing |
| `Life expectancy` | Life expectancy at birth (years) |
| `Adult Mortality` | Adult mortality rate per 1000 population |
| `Infant deaths` | Infant deaths per 1000 live births |
| `Under-five deaths` | Under-five mortality per 1000 live births |
| `GDP` | Gross Domestic Product per capita (USD) |
| `Schooling` | Average years of schooling |
| `BMI` | Average Body Mass Index |
| `Alcohol` | Alcohol consumption per capita (litres) |
| `percentage expenditure` | Health expenditure as % of GDP |
| `Population` | Country population |

**Source:** Modelled after the [Kaggle WHO Life Expectancy Dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations, log transforms |
| **Matplotlib** | Base plotting engine |
| **Seaborn** | Statistical visualisations (heatmaps, KDE, regression) |
| **Scikit-learn** | StandardScaler, VarianceThreshold, LinearRegression |
| **yData Profiling** | Automated profiling (install separately) |

---

## 💡 Key Insights

1. **Schooling** is the strongest positive predictor of life expectancy (Pearson r ≈ 0.75).
2. **Adult Mortality Rate** is the dominant negative predictor.
3. **GDP** follows a logarithmic relationship — wealthier nations gain diminishing returns.
4. Developing countries improved faster (**+2.5 yrs/decade**) vs Developed (**+1.5 yrs**).
5. **Infant deaths** and **Under-five deaths** are nearly collinear — use only one in models.
6. **Alcohol consumption** shows a mild *positive* correlation (confounded by wealthier nations drinking more).
7. Linear Regression with 10 standardised features achieves **R² ≈ 0.85**.

---

## 🚀 How to Run

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `ydata-profiling` requires an internet connection to install.  
> The project runs fully without it — the HTML report is generated natively.

### 2 — Run the full pipeline
```bash
python run_eda.py
```

This will:
- ✅ Generate the dataset (`data/life_expectancy.csv`)
- ✅ Clean the data and save `data/life_expectancy_clean.csv`
- ✅ Generate all 9 visualisation plots → `/reports`
- ✅ Train a Linear Regression model and print R² / RMSE
- ✅ Build the standalone HTML report → `reports/eda_profile.html`

### 3 — Build the notebook
```bash
python build_notebook.py
jupyter notebook notebooks/eda_analysis.ipynb
```

### 4 — Open the report
```bash
open reports/eda_profile.html   # macOS
xdg-open reports/eda_profile.html  # Linux
start reports/eda_profile.html  # Windows
```

---

## 📁 Output Files

| File | Description |
|---|---|
| `reports/eda_profile.html` | Self-contained interactive HTML report |
| `reports/01_*.png … 09_*.png` | Individual plot images |
| `data/life_expectancy_clean.csv` | Cleaned dataset ready for modelling |

---

## 🔬 Methodology

### Data Cleaning
- **Duplicates:** Exact duplicate rows removed
- **Missing values:** Group-wise median imputation (by Status), global median fallback
- **Outliers:** Winsorisation at Q1 − 3×IQR and Q3 + 3×IQR (conservative)

### Feature Engineering
- `log_GDP` — log(GDP+1) to linearise the skewed distribution
- `mortality_ratio` — Adult Mortality / (Infant deaths + 1)
- `GDP_per_schooling` — Economic efficiency proxy
- StandardScaler applied before modelling
- VarianceThreshold removes near-constant features

### Model
- **Algorithm:** Ordinary Least Squares (Linear Regression)
- **Split:** 80% train / 20% test (random_state=42)
- **Metrics:** R² and RMSE

---

## 📈 Next Steps

- [ ] Try **Random Forest** / **Gradient Boosting** for non-linear relationships
- [ ] Add **country fixed effects** using one-hot encoding
- [ ] Perform **time-series analysis** per country
- [ ] Build an interactive **Plotly / Streamlit dashboard**
- [ ] Run **PCA** on scaled features for dimensionality reduction

---

## 👤 Author

Built as an industry-quality student data science project using Python.  
Feel free to fork, extend, and adapt!

---

*License: MIT*
