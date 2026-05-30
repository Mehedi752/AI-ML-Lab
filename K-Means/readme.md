# 🧠 K-Means Clustering – Customer Segmentation Project

## 📌 Project Overview
This project implements **K-Means Clustering** on the **Mall Customers Dataset** to perform customer segmentation based on purchasing behavior.  
The model groups customers into meaningful clusters using features such as **Annual Income** and **Spending Score**.

Additionally, the trained model is tested on **real-world custom data (10 individuals)** to demonstrate practical clustering applications.

---

## 📂 Dataset Information

### Standard Dataset:
- Mall Customers Dataset
- Features:
  - CustomerID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

### Custom Dataset:
- Contains 10 real-world survey records
- Features:
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

---

## ⚙️ Technologies Used
- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🧪 Workflow Steps

1. Data Loading (Automatic via GitHub)
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing (Cleaning + Scaling)
4. Elbow Method for Optimal K Selection
5. K-Means Model Training
6. Model Saving using Joblib
7. Custom Data Prediction
8. Cluster Visualization

---

## 📊 Optimal Number of Clusters

The Elbow Method was used to determine the optimal value of K.

- Optimal K selected: **5**

---

## 📈 Final Visualizations

### 1. Elbow Method Plot
Shows the relationship between **number of clusters (K)** and **WCSS (Within-Cluster Sum of Squares)**.

📌 Interpretation:
- The "elbow point" indicates the optimal number of clusters.

---

### 2. Cluster Visualization (Final Result)

- Customers are grouped into 5 clusters
- Each cluster is represented by a different color
- Cluster centroids are marked with an "X"

📌 This visualization helps understand customer behavior patterns clearly.

---

### 3. Custom Data Prediction Table

A table showing:
- Age
- Annual Income
- Spending Score
- Assigned Cluster ID

📌 This demonstrates real-world usage of the trained model.

---

## 🧠 Cluster Interpretation

### 🔵 Cluster 0 – High Income, High Spending
- Premium customers
- High purchasing power
- Ideal target for luxury marketing campaigns

---

### 🟢 Cluster 1 – High Income, Low Spending
- Wealthy but cautious customers
- Prefer saving rather than spending
- Require premium offers or incentives

---

### 🟡 Cluster 2 – Medium Income, Medium Spending
- Balanced customers
- Stable purchasing behavior
- Standard marketing approach works best

---

### 🔴 Cluster 3 – Low Income, High Spending
- Impulsive buyers
- Spend more than income level suggests
- Respond well to discounts and promotions

---

### 🟣 Cluster 4 – Low Income, Low Spending
- Budget-conscious customers
- Least spending behavior
- Require cost-effective marketing strategies

---

## 💾 Model File

The trained K-Means model is saved using Joblib:
