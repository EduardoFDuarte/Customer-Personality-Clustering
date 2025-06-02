# -*- coding: utf-8 -*-
"""
# Customer Segmentation: Supermarket Sales Analysis

This notebook walks through an in-depth customer segmentation of a supermarket sales dataset.  
It includes:
- **RFM feature engineering** - Stands for Recency, Frequency, and Monetary value. It’s a method used to analyze customer behavior and segment customers based on their purchasing habits.
- **Extensive exploratory data analysis**
- **Dimensionality reduction (PCA)** - Principal Component Analysis projection, is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It achieves this by identifying the principal components, which are orthogonal axes that capture the most variance in the original data.
- **Cluster-selection diagnostics (Elbow, silhouette)**
- **K-Means & DBSCAN clustering**
- **Detailed cluster profiling**

Remarks throughout explain the expected outputs.

## 1. Importing Libraries

We import standard data-science libraries, clustering tools, and evaluation metrics.
"""

import pandas as pd
import numpy as np


from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

import plotly.express as px
import plotly.figure_factory as ff

# For elbow & silhouette diagnostics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Pandas display options
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

"""## 2. Loading Data

"""

df = pd.read_csv("marketing_campaign.csv", sep="\t")
print("Shape:", df.shape)
df.info()
df.head()

"""## 3. Data Cleaning

"""

# Missing values
print(df.isna().sum())

# Drop duplicates
df = df.drop_duplicates()
print("After deduplication:", df.shape)

# Fill missing values with mean for numerical columns in numeric columns only
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
print("After filling missing values:", df.isna().sum().sum())

df.isna().sum()

# Sanity check Recency (should be non-negative)
print("Recency: min =", df['Recency'].min(), "max =", df['Recency'].max())

"""## 4. RFM Feature Engineering

"""

# Since each row is one customer, compute:
rfm = pd.DataFrame()
rfm['Recency']   = df['Recency']
rfm['Frequency'] = (
    df['NumDealsPurchases']
  + df['NumWebPurchases']
  + df['NumCatalogPurchases']
  + df['NumStorePurchases']
)
rfm['Monetary']  = (
    df['MntWines']
  + df['MntFruits']
  + df['MntMeatProducts']
  + df['MntFishProducts']
  + df['MntSweetProducts']
  + df['MntGoldProds']
)
# Log‐transform Monetary to reduce skew
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

print(rfm.describe().T)

"""## 5. Data Preprocessing

"""

scaler = StandardScaler()
rfm_scaled = pd.DataFrame(
    scaler.fit_transform(rfm[['Recency','Frequency','Monetary_log']]),
    columns=['Recency','Frequency','Monetary_log']
)
print(rfm_scaled.describe().T)

"""## 6. Exploratory Data Analysis
### 6.1 RFM Distributions

"""

fig = px.histogram(rfm, x='Recency', nbins=30, title='Recency Distribution')
fig.show()

fig = px.histogram(rfm, x='Frequency', nbins=30, title='Frequency Distribution')
fig.show()

fig = px.histogram(rfm, x='Monetary', nbins=30, title='Monetary Distribution')
fig.show()

"""### 6.2 RFM Correlation Heatmap

"""

corr = rfm[['Recency','Frequency','Monetary']].corr().values
labels = ['Recency','Frequency','Monetary']
fig = ff.create_annotated_heatmap(
    z=corr, x=labels, y=labels,
    colorscale='Viridis'
)
fig.update_layout(title='Correlation Matrix of RFM')
fig.show()

"""### 6.3 Pairwise Relationships (Scaled RFM)

"""

import plotly.graph_objects as go

fig = go.Figure(
    data=go.Splom(
        dimensions=[
            dict(label='Recency',       values=rfm_scaled['Recency']),
            dict(label='Frequency',     values=rfm_scaled['Frequency']),
            dict(label='Monetary_log',  values=rfm_scaled['Monetary_log']),
        ],
        diagonal=dict(visible=False),
        marker=dict(size=4, color='steelblue', opacity=0.6)
    )
)
fig.update_layout(
    title='Pairwise Scatter Matrix of Scaled RFM (go.Splom)',
    width=700,
    height=700
)
fig.show()

"""## 7. Dimensionality Reduction (PCA)

"""

pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)

fig = px.scatter(
    x=rfm_pca[:,0], y=rfm_pca[:,1],
    title='PCA Projection of RFM',
    labels={'x':'PC1','y':'PC2'}
)
fig.show()

"""## 8. Determining Optimal K for K-Means

"""

inertias = []
silhouettes = []
K_range = range(2,11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42).fit(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, km.labels_))

fig = px.line(x=list(K_range), y=inertias,
              markers=True, title='Elbow Plot (Inertia vs. K)',
              labels={'x':'k','y':'Inertia'})
fig.show()

fig = px.line(x=list(K_range), y=silhouettes,
              markers=True, title='Silhouette Score vs. K',
              labels={'x':'k','y':'Silhouette'})
fig.show()

"""## 9. Clustering

"""

# 9.1 K-Means with chosen k (e.g. 4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
rfm['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)

# 9.2 DBSCAN for comparison
dbscan = DBSCAN(eps=1.0, min_samples=5)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled)

"""### 9.1 K-Means Clusters in PCA Space

"""

fig = px.scatter(
    x=rfm_pca[:,0], y=rfm_pca[:,1],
    color=rfm['KMeans_Cluster'].astype(str),
    title='K-Means Clusters (PCA-reduced)',
    labels={'x':'PC1','y':'PC2','color':'Cluster'}
)
fig.show()

"""### 9.2 DBSCAN Clusters in PCA Space

"""

fig = px.scatter(
    x=rfm_pca[:,0], y=rfm_pca[:,1],
    color=rfm['DBSCAN_Cluster'].astype(str),
    title='DBSCAN Clusters (PCA-reduced)',
    labels={'x':'PC1','y':'PC2','color':'Cluster'}
)
fig.show()

"""## 10. Evaluating Clustering Quality

"""

for name, labels in [
    ('KMeans', rfm['KMeans_Cluster']),
    ('DBSCAN', rfm['DBSCAN_Cluster'])
]:
    unique = set(labels)
    if len(unique) > 1 and (name=='KMeans' or -1 not in unique):
        sil = silhouette_score(rfm_scaled, labels)
        db  = davies_bouldin_score(rfm_scaled, labels)
        print(f"{name:>7} | silhouette = {sil:.3f} | Davies-Bouldin = {db:.3f}")

"""## 11. Cluster Profiling

"""

# 1. Make sure cluster labels live in both rfm and df
# (rfm already has KMeans_Cluster & DBSCAN_Cluster)
df['KMeans_Cluster']  = rfm['KMeans_Cluster'].values
df['DBSCAN_Cluster']  = rfm['DBSCAN_Cluster'].values

# 2. Profile RFM metrics (mean per cluster)
profile_rfm = (
    rfm
    .groupby('KMeans_Cluster')[['Recency','Frequency','Monetary']]
    .mean()
    .rename(columns={
        'Recency':'Avg_Recency',
        'Frequency':'Avg_Frequency',
        'Monetary':'Avg_Monetary'
    })
)
print("🔹 K-Means RFM Cluster Profile")
print(profile_rfm)

# 3. Profile key numeric demographics/behavior from original df
demo_profile = (
    df
    .groupby('KMeans_Cluster')[['Income','Kidhome','Teenhome']]
    .mean()
    .rename(columns={
        'Income':'Avg_Income',
        'Kidhome':'Avg_Num_Kids',
        'Teenhome':'Avg_Num_Teens'
    })
)
print("🔹 K-Means Demographic & Behavior Profile")
print(demo_profile)

# 4. Categorical distributions (e.g. Education level)
edu_dist = (
    df
    .groupby('KMeans_Cluster')['Education']
    .value_counts(normalize=True)
    .unstack()
)
print("🔹 K-Means Education Distribution (%)")
print(edu_dist)

"""## 12. Conclusion

- **Cluster 0**: Low Recency, High Frequency, High Monetary → **VIPs**  
  Propose: exclusive early access to new products, tiered loyalty rewards, personalized premium offers.

- **Cluster 1**: High Recency, Low Frequency, Low Monetary → **At-risk/Lapsing Customers**  
  Propose: “We miss you” re-engagement emails, time-limited discounts, win-back vouchers.

- **Cluster 2**: Moderate Recency, Moderate Frequency, Moderate Monetary → **Growth Segment**  
  Propose: bundle or “frequently bought together” promotions, cross-sell campaigns, incremental-spend incentives.

- **Cluster 3**: Low Monetary, High Frequency → **Deal-Seeking Frequent Shoppers**  
  Propose: flash sales on higher-margin items, targeted coupons on premium products, “only for you” upsell offers.  

Propose targeted campaigns accordingly.  

"""