"""Store/department segmentation via clustering."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Feature engineering & clustering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame):
    agg = (
        df.groupby("Store")["Weekly_Sales"]
        .agg(["mean", "std", "max", "min"])
        .rename(columns={"mean": "sales_mean", "std": "sales_std"})
    )
    agg["size"] = df.groupby("Store")["Size"].first()
    return agg.reset_index()


def cluster_stores(df: pd.DataFrame, k: int = 4):
    features = df.drop("Store", axis=1)
    X_scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    df["Cluster"] = model.labels_
    score = silhouette_score(X_scaled, model.labels_)
    return df, model, score