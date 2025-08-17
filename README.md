# 🏨 Hotel Guest Preferences Clustering

This project applies **Unsupervised Machine Learning (K-Means Clustering)** 
to group hotel guests based on their **stay duration, spending behavior, and booking preferences**.


## 📊 Steps Followed
1. Data Cleaning & Feature Engineering  
2. One-Hot Encoding + Scaling  
3. PCA for Dimensionality Reduction  
4. K-Means Clustering  
5. Elbow Method & Silhouette Score for evaluation  
6. Visualization of Clusters  
7. Insights & Recommendations  


## 🧑‍💻 Model Used
- **K-Means Clustering** (main model)  
- PCA (for visualization & dimensionality reduction)  
- Elbow Method + Silhouette Score (for evaluation)  


## 📈 Visualizations

### Elbow Method
![Elbow](charts/elbow_method.png)

### Guest Clusters (PCA 2D View)
![Clusters](charts/pca_clusters.png)

### Average Spending by Cluster
![Spending](charts/cluster_spending.png)

---

## 🔍 Insights
- Guests can be grouped into clusters like **Luxury Seekers**, **Budget Travelers**, etc.  
- High-spending guests usually stay longer and prefer premium room types.  
- Budget guests prefer short stays with minimum services.  


## 🚀 Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn, Seaborn, Matplotlib  
- VS Code  

## 📂 Folder Structure
hotel-guest-clustering/
│── project.py
│── hotel_bookings.csv
│── charts/
│ ├── elbow_method.png
│ ├── pca_clusters.png
│ ├── cluster_spending.png
│── README.md

PROJECT BY YASHVI VERMA
