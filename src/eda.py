"""Exploratory Data Analysis visualizations."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parents[1] / "output" / "visuals"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

# ---------------------------------------------------------------------------
# Generic Plot Helpers
# ---------------------------------------------------------------------------

def save(fig, name: str):
    fig.savefig(PLOT_DIR / name, bbox_inches="tight", dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Specific EDA plots
# ---------------------------------------------------------------------------

def sales_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Weekly_Sales"], bins=100, ax=ax)
    ax.set_title("Sales Distribution")
    save(fig, "sales_distribution.png")


def time_trend(df: pd.DataFrame):
    monthly = (
        df.assign(Month=lambda x: x["Date"].dt.to_period("M"))
        .groupby("Month")["Weekly_Sales"].sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=monthly, x="Month", y="Weekly_Sales", ax=ax)
    ax.set_title("Total Monthly Sales")
    save(fig, "monthly_sales_trend.png")