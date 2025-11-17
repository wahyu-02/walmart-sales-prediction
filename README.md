# 🛍️ Walmart Sales Prediction with XGBoost

This project predicts weekly sales for Walmart departments using machine learning. It analyzes the impact of weather, holidays, and store-level data, culminating in an interactive dashboard built with Streamlit.

### 🚀 [View the Live Streamlit App!](YOUR_STREAMLIT_APP_URL)

*(Go to your Streamlit app, copy its URL, and paste it above)*

---

### ## Project Overview

The goal was to build an end-to-end regression model to forecast sales. This involved:
* **Data Cleaning:** Merging and cleaning store, feature, and sales data.
* **Exploratory Data Analysis (EDA):** Visualizing seasonality and the impact of temperature and holidays on sales.
* **Feature Engineering:** Creating lag features (`Sales_Lag_1_Week`) and rolling averages (`Sales_Avg_4_Weeks`) to capture time-series trends.
* **Model Training:** Comparing Linear Regression, Random Forest, and **XGBoost**, with XGBoost performing the best.

### ## Key Results
The final XGBoost model achieved the following performance on the test set (final 6 months of data):

| Metric | Score | Description |
| :--- | :--- | :--- |
| **RMSE** | `$3,038.51` | The average prediction error in dollars. |
| **MAE** | `$1,410.71` | The average absolute error in dollars. |

### ## Technology Stack
* **Python**
* **Pandas & NumPy** for data manipulation
* **Scikit-learn** for modeling pipeline
* **XGBoost** for the final regression model
* **Plotly** for interactive charts
* **Streamlit** for the interactive web dashboard
