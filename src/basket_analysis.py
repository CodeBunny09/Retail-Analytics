"""Market basket analysis at department level."""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def prepare_baskets(df: pd.DataFrame):
    pivot = (
        df.pivot_table(index=["Store", "Date"], columns="Dept", values="Weekly_Sales", aggfunc="sum")
        .fillna(0)
    )
    # Convert to binary (presence/absence of significant sales)
    return (pivot > 0).astype(int)


def run_apriori(basket: pd.DataFrame, min_support: float = 0.01):
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1)
    return rules.sort_values("lift", ascending=False)