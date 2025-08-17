# ===========================================
# HOTEL GUEST PREFERENCES CLUSTERING PROJECT
# ===========================================

# PHASE 1 & 2: Data Loading & Preparation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---- Load dataset (change path if needed)
df = pd.read_csv("hotel_bookings.csv")

# Select relevant columns
cols = [
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'meal',
    'market_segment',
    'adr',  # Average Daily Rate
    'reserved_room_type',
    'customer_type'
]
df = df[cols]

# Handle missing values (if any)
df = df.dropna()

# Create new features
df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_spending'] = df['adr'] * df['total_stay_duration']

# Drop original stay columns (optional)
df = df.drop(['stays_in_weekend_nights', 'stays_in_week_nights'], axis=1)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['meal', 'market_segment', 'reserved_room_type', 'customer_type'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# PHASE 3: PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

print("Explained variance ratio by 2 PCA components:", pca.explained_variance_ratio_)

# PHASE 4: Finding Optimal K (Elbow Method)
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

# Create folder for charts
os.makedirs("charts", exist_ok=True)

# Save Elbow Method plot
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.savefig("charts/elbow_method.png", dpi=300, bbox_inches='tight')
plt.close()

# Choose K (adjust after checking elbow plot)
optimal_k = 4  

# PHASE 5: Clustering with KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_data)

# Silhouette Score
sil_score = silhouette_score(pca_data, clusters)
print(f"Silhouette Score for k={optimal_k}: {sil_score:.3f}")

# Add cluster labels to dataframe
df['Cluster'] = clusters

# PHASE 6: Visualization
# PCA Scatter Plot with centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='Set2', alpha=0.4, s=30)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("Hotel Guest Clusters (PCA 2D View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.savefig("charts/pca_clusters.png", dpi=300, bbox_inches='tight')
plt.close()

# Cluster-wise average spending
cluster_spend = df.groupby('Cluster')['total_spending'].mean().sort_values(ascending=False)
cluster_spend.plot(kind='bar', color='skyblue')
plt.title("Average Spending per Cluster")
plt.ylabel("Average Spending")
plt.savefig("charts/cluster_spending.png", dpi=300, bbox_inches='tight')
plt.close()

# PHASE 7: Insights
print("\nCluster Insights:")
for c in df['Cluster'].unique():
    sub = df[df['Cluster'] == c]
    print(f"\nCluster {c}:")
    print(f"  Avg Stay Duration: {sub['total_stay_duration'].mean():.1f} nights")
    print(f"  Avg Spending: {sub['total_spending'].mean():.2f}")
    print(f"  Most Common Meal: {sub['meal'].mode()[0]}")
    print(f"  Most Common Booking Channel: {sub['market_segment'].mode()[0]}")

print("\n✅ All plots saved in 'charts/' folder.")


