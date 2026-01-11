Sales Forecasting & Analytics Dashboard

An end-to-end Machine Learning project that predicts future sales and visualizes business insights through an interactive web dashboard built with Flask and Plotly.
This project demonstrates the complete ML lifecycle:
Data preprocessing → Feature engineering → Model training → API deployment → Interactive visualization.

FEATURES

-Sales prediction using a trained Random Forest model
-Monthly sales trend visualization
-Revenue analysis by product category
-Customer demographic insights
-Model performance evaluation (Actual vs Predicted)
-Business KPIs:
  -Total Revenue
  -Average Order Value
  -Top Performing Category
  -Total Customers
-Flask REST API for real-time predictions
-Interactive dashboard built with Plotly and HTML/CSS

TECH STACK

-Python
-Pandas, NumPy
-Scikit-learn
-Flask
-Plotly
-HTML, CSS
-Git & GitHub

SYSTEM ARCHITECTURE

Retail Sales Dataset
→ Data Cleaning & Feature Engineering
→ Random Forest Regression Model
→ Flask REST API (/predict)
→ Interactive Web Dashboard (Plotly + HTML/CSS)

PROJECT STRUCTURE

sales-forcasting-ml/
│── app.py
│── sales_model.pkl
│── model_features.pkl
│── data/
│ └── retail_sales_dataset.csv
│── templates/
│ ├── index.html
│ ├── trends.html
│ ├── categories.html
│ └── performance.html
│── static/
│ └── style.css
│── requirements.txt
│── README.md

HOW TO RUN LOCALLY

-Clone the repository
 git clone https://github.com/Keertan06/sales-forcasting-ml.git
 cd sales-forcasting-ml

-Create virtual environment
 python -m venv .venv
 source .venv/bin/activate (Mac/Linux)
 .venv\Scripts\activate (Windows)

-Install dependencies
 pip install -r requirements.txt

-Run the Flask app
 python app.py

-Open in browser
 http://127.0.0.1:5000

DASHBOARD PAGES

-Prediction – Enter customer and product details to predict sales
-Trends – Monthly sales time-series visualization
-Categories – Category-wise and gender-wise revenue insights
-Performance – Actual vs Predicted model evaluation

FUTURE ENHANCEMENTS

-Date range filters for trends
-Dynamic KPI updates
-Feature importance & SHAP explanations
-Model confidence intervals

AUTHOR

Keertan Kumar
Aspiring Data Scientist / ML Engineer
GitHub: https://github.com/Keertan06

If you like this project, give it a star on GitHub and feel free to fork or contribute.
