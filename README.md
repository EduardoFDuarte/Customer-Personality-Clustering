# Customer Segmentation: Supermarket Sales Analysis

## Project Overview

This project provides an extensive analysis of customer segmentation using a supermarket sales dataset. It leverages customer behavior analytics to form actionable insights that help target customer groups more effectively.

## Key Objectives

* **RFM Feature Engineering**: Recency, Frequency, Monetary analysis to segment customer purchasing habits.
* **Exploratory Data Analysis (EDA)**: Understanding data distributions, correlations, and customer behavior patterns.
* **Dimensionality Reduction**: Applying Principal Component Analysis (PCA) for visualization and analysis simplification.
* **Clustering**: Utilizing K-Means and DBSCAN to identify distinct customer segments.
* **Cluster Profiling**: Detailed profiling for strategic insights into each customer segment.

## Libraries Used

* **Data Manipulation**: Pandas, NumPy
* **Preprocessing**: Scikit-learn (StandardScaler)
* **Dimensionality Reduction**: Scikit-learn (PCA)
* **Clustering**: Scikit-learn (KMeans, DBSCAN)
* **Evaluation Metrics**: Scikit-learn (Silhouette Score, Davies-Bouldin Index)
* **Visualization**: Plotly, Yellowbrick

## Data

The analysis uses a dataset (`marketing_campaign.csv`) containing various customer attributes and transaction history, facilitating the computation of RFM metrics.

## Methodology

### Data Cleaning

* Handling missing values and duplicates to ensure data integrity.

### Feature Engineering

* Calculating Recency, Frequency, and Monetary values.
* Log-transforming Monetary to address skewness.

### Exploratory Analysis

* Visual distribution and correlation checks.

### Dimensionality Reduction

* PCA to visualize high-dimensional data in two dimensions.

### Clustering

* Optimal K determination using elbow and silhouette methods.
* K-Means clustering for segment identification.
* DBSCAN clustering for alternative insights.

### Cluster Profiling

* Demographic, behavioral, and educational profiling to create actionable customer segments.

## Results and Recommendations

Identified four main customer segments:

1. **VIP Customers**: Loyal, frequent, high spenders—ideal for premium and loyalty rewards.
2. **At-risk Customers**: Recent interactions with declining frequency—targeted re-engagement campaigns.
3. **Growth Segment**: Potential for increased value—recommend bundled promotions.
4. **Deal-Seeking Shoppers**: Frequent buyers of lower-value products—opportunities for upselling higher-margin items.

## Usage

Run the provided Python scripts or Jupyter Notebook (`fpcode.py` or provided notebooks) for complete analysis execution.

## Further Improvements

* Integration with external customer databases for enhanced insights.
* Implementation of advanced machine learning techniques for dynamic segmentation.
