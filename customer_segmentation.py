import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Income":[15,16,17,45,46,48,80,82,85],
    "Spending":[39,42,35,65,70,68,90,88,95]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

print(df)

plt.scatter(
    df["Income"],
    df["Spending"],
    c=df["Cluster"]
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()