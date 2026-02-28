"""
src/visualization.py
---------------------
All Matplotlib / Seaborn plots for the EDA.
Each function:
  - Accepts a cleaned DataFrame
  - Shows AND saves the figure to reports/
  - Returns the figure object
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts & CI)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ─── Global Aesthetics ────────────────────────────────────────────────────────
PALETTE      = "viridis"
STATUS_PAL   = {"Developed": "#2196F3", "Developing": "#FF5722"}
REPORTS_DIR  = "reports"
DPI          = 150
FIG_SIZE_STD = (10, 6)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
os.makedirs(REPORTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  ✔ Saved → {path}")


# ─── 1. Distribution of Life Expectancy ──────────────────────────────────────

def plot_life_expectancy_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE of life expectancy, split by development status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution of Life Expectancy", fontsize=16, fontweight="bold")

    # Overall
    sns.histplot(df["Life expectancy"], bins=30, kde=True,
                 color="#4C72B0", ax=axes[0])
    axes[0].set_title("Overall Distribution")
    axes[0].set_xlabel("Life Expectancy (years)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df["Life expectancy"].mean(), color="red",
                    linestyle="--", label=f"Mean: {df['Life expectancy'].mean():.1f}")
    axes[0].axvline(df["Life expectancy"].median(), color="orange",
                    linestyle="--", label=f"Median: {df['Life expectancy'].median():.1f}")
    axes[0].legend()

    # By Status
    for status, color in STATUS_PAL.items():
        subset = df[df["Status"] == status]["Life expectancy"]
        sns.kdeplot(subset, ax=axes[1], label=status, color=color, fill=True, alpha=0.35)
    axes[1].set_title("By Development Status (KDE)")
    axes[1].set_xlabel("Life Expectancy (years)")
    axes[1].set_ylabel("Density")
    axes[1].legend(title="Status")

    plt.tight_layout()
    _save(fig, "01_life_expectancy_distribution.png")
    return fig


# ─── 2. Life Expectancy Trends Over Years ─────────────────────────────────────

def plot_trend_over_years(df: pd.DataFrame) -> plt.Figure:
    """Line plot of mean life expectancy per year, by development status."""
    trend = (
        df.groupby(["Year", "Status"])["Life expectancy"]
        .mean()
        .reset_index()
    )
    overall = df.groupby("Year")["Life expectancy"].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIG_SIZE_STD)
    for status, color in STATUS_PAL.items():
        subset = trend[trend["Status"] == status]
        ax.plot(subset["Year"], subset["Life expectancy"],
                marker="o", markersize=4, label=status, color=color, linewidth=2)

    ax.plot(overall["Year"], overall["Life expectancy"],
            marker="s", markersize=4, label="Global Average",
            color="black", linewidth=2.5, linestyle="--")

    ax.set_title("Global Life Expectancy Trends (2000–2015)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Life Expectancy (years)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(title="Status")
    plt.tight_layout()
    _save(fig, "02_trend_over_years.png")
    return fig


# ─── 3. GDP vs Life Expectancy (Regression Plot) ─────────────────────────────

def plot_gdp_vs_life_expectancy(df: pd.DataFrame) -> plt.Figure:
    """Scatter + regression line for GDP vs Life Expectancy."""
    df_plot = df.dropna(subset=["GDP", "Life expectancy"]).copy()
    df_plot["log_GDP"] = np.log1p(df_plot["GDP"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GDP vs Life Expectancy", fontsize=16, fontweight="bold")

    # Raw GDP
    for status, color in STATUS_PAL.items():
        sub = df_plot[df_plot["Status"] == status]
        axes[0].scatter(sub["GDP"], sub["Life expectancy"],
                        alpha=0.4, color=color, label=status, s=15)
    axes[0].set_xlabel("GDP (USD)")
    axes[0].set_ylabel("Life Expectancy (years)")
    axes[0].set_title("Raw GDP")
    axes[0].legend(title="Status")

    # Log GDP with regression
    sns.regplot(data=df_plot, x="log_GDP", y="Life expectancy",
                scatter_kws={"alpha": 0.25, "s": 12, "color": "#607D8B"},
                line_kws={"color": "crimson", "linewidth": 2},
                ax=axes[1])
    axes[1].set_xlabel("log(GDP + 1)")
    axes[1].set_ylabel("Life Expectancy (years)")
    axes[1].set_title("log(GDP) — Regression Fit")

    plt.tight_layout()
    _save(fig, "03_gdp_vs_life_expectancy.png")
    return fig


# ─── 4. Schooling vs Life Expectancy ─────────────────────────────────────────

def plot_schooling_vs_life_expectancy(df: pd.DataFrame) -> plt.Figure:
    """Scatter + regression for Schooling vs Life Expectancy."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_STD)

    for status, color in STATUS_PAL.items():
        sub = df[df["Status"] == status]
        ax.scatter(sub["Schooling"], sub["Life expectancy"],
                   alpha=0.4, color=color, label=status, s=15)

    # Overall regression line
    sns.regplot(data=df, x="Schooling", y="Life expectancy",
                scatter=False,
                line_kws={"color": "black", "linewidth": 2, "linestyle": "--"},
                ax=ax)

    ax.set_title("Schooling vs Life Expectancy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Years of Schooling")
    ax.set_ylabel("Life Expectancy (years)")
    ax.legend(title="Status")
    plt.tight_layout()
    _save(fig, "04_schooling_vs_life_expectancy.png")
    return fig


# ─── 5. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Annotated heatmap of Pearson correlations among numeric features."""
    numeric_cols = [
    "Life expectancy", "Adult Mortality", "infant deaths",
    "under-five deaths", "GDP", "Schooling", "BMI",
    "Alcohol", "percentage expenditure", "Population"
]
    corr = df[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle masked
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 9}, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png")
    return fig


# ─── 6. Region/Status-wise Comparison ────────────────────────────────────────

def plot_status_comparison(df: pd.DataFrame) -> plt.Figure:
    """Box plots comparing key health indicators by development status."""
    metrics = [
        ("Life expectancy",       "Life Expectancy (years)"),
        ("Adult Mortality",       "Adult Mortality Rate"),
        ("GDP",                   "GDP (USD)"),
        ("Schooling",             "Years of Schooling"),
        ("percentage expenditure","Health Expenditure (%)"),
        ("BMI",                   "Average BMI"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Health & Economic Indicators by Development Status",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics):
        sns.boxplot(
            data=df, x="Status", y=col,
            palette=STATUS_PAL, width=0.5,
            flierprops={"marker": ".", "markersize": 3}, ax=ax
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(label)

    plt.tight_layout()
    _save(fig, "06_status_comparison.png")
    return fig


# ─── 7. Bonus: Pairplot of Key Features ──────────────────────────────────────

def plot_pairplot(df: pd.DataFrame) -> plt.Figure:
    """Seaborn pairplot of the most correlated features."""
    cols = ["Life expectancy", "GDP", "Schooling", "Adult Mortality", "BMI", "Status"]
    sample = df[cols].dropna().sample(min(400, len(df)), random_state=42)

    g = sns.pairplot(
        sample, hue="Status", palette=STATUS_PAL,
        diag_kind="kde", plot_kws={"alpha": 0.35, "s": 15},
        corner=True
    )
    g.figure.suptitle("Pairplot — Key Features", y=1.01, fontsize=14, fontweight="bold")
    _save(g.figure, "07_pairplot.png")
    return g.figure


# ─── 8. Top / Bottom Countries ───────────────────────────────────────────────

def plot_top_bottom_countries(df: pd.DataFrame, year: int = 2015) -> plt.Figure:
    """Horizontal bar chart of top & bottom 10 countries for a given year."""
    yr_df = (
        df[df["Year"] == year]
        .groupby("Country")["Life expectancy"]
        .mean()
        .sort_values()
    )
    bottom10 = yr_df.head(10)
    top10    = yr_df.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Life Expectancy by Country ({year})", fontsize=15, fontweight="bold")

    bottom10.plot(kind="barh", ax=axes[0], color="#EF5350")
    axes[0].set_title("Bottom 10 Countries")
    axes[0].set_xlabel("Life Expectancy (years)")

    top10.plot(kind="barh", ax=axes[1], color="#66BB6A")
    axes[1].set_title("Top 10 Countries")
    axes[1].set_xlabel("Life Expectancy (years)")

    plt.tight_layout()
    _save(fig, "08_top_bottom_countries.png")
    return fig


# ─── Master function ──────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame) -> None:
    """Generate and save all EDA visualizations."""
    print("\n📊 Generating visualizations …")
    plot_life_expectancy_distribution(df)
    plot_trend_over_years(df)
    plot_gdp_vs_life_expectancy(df)
    plot_schooling_vs_life_expectancy(df)
    plot_correlation_heatmap(df)
    plot_status_comparison(df)
    plot_pairplot(df)
    plot_top_bottom_countries(df)
    print("✅ All plots saved to /reports\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_cleaning import clean_pipeline

    df = clean_pipeline(save=False)
    generate_all_plots(df)
