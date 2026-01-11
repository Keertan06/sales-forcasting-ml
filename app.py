from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load trained model and schema
model = joblib.load("sales_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Load dataset
df = pd.read_csv("data/retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

def create_plots():
    # Monthly Sales Trend
    monthly = df.groupby(df['Date'].dt.to_period("M"))['Total Amount'].sum().reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    fig_trend = px.line(monthly, x='Date', y='Total Amount', title="Monthly Sales Trend")

    # Category Revenue
    cat = df.groupby('Product Category')['Total Amount'].sum().reset_index()
    fig_cat = px.bar(cat, x='Product Category', y='Total Amount', title="Revenue by Category")

    # Gender Distribution
    gender = df.groupby('Gender')['Total Amount'].sum().reset_index()
    fig_gender = px.pie(gender, names='Gender', values='Total Amount', title="Spending by Gender")

    # Actual vs Predicted
    df_fe = df.copy()
    df_fe['Year'] = df_fe['Date'].dt.year
    df_fe['Month'] = df_fe['Date'].dt.month
    df_fe['Day'] = df_fe['Date'].dt.day
    df_fe = pd.get_dummies(df_fe, columns=['Gender', 'Product Category'], drop_first=True)
    df_fe = df_fe.drop(['Transaction ID', 'Customer ID', 'Date'], axis=1)

    X = df_fe.drop(['Total Amount', 'Quantity', 'Price per Unit'], axis=1)
    y = df_fe['Total Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_local = RandomForestRegressor(n_estimators=100, random_state=42)
    model_local.fit(X_train, y_train)
    y_pred = model_local.predict(X_test)

    perf_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig_perf = px.scatter(perf_df, x='Actual', y='Predicted', title="Actual vs Predicted Sales")

    return (
        pio.to_html(fig_trend, full_html=False),
        pio.to_html(fig_cat, full_html=False),
        pio.to_html(fig_gender, full_html=False),
        pio.to_html(fig_perf, full_html=False)
    )
def compute_kpis():
    total_revenue = round(df['Total Amount'].sum(), 2)
    avg_order = round(df['Total Amount'].mean(), 2)
    top_category = df.groupby('Product Category')['Total Amount'].sum().idxmax()
    total_customers = df['Customer ID'].nunique()

    return {
        "total_revenue": total_revenue,
        "avg_order": avg_order,
        "top_category": top_category,
        "total_customers": total_customers
    }

# ---------- Routes ----------

@app.route("/")
def index():
    kpis = compute_kpis()
    return render_template("index.html", kpis=kpis)

@app.route("/trends")
def trends():
    trend_plot, _, _, _ = create_plots()
    kpis = compute_kpis()
    return render_template("trends.html", trend_plot=trend_plot, kpis=kpis)

@app.route("/categories")
def categories():
    _, cat_plot, gender_plot, _ = create_plots()
    kpis = compute_kpis()
    return render_template("categories.html", cat_plot=cat_plot, gender_plot=gender_plot, kpis=kpis)

@app.route("/performance")
def performance():
    _, _, _, performance_plot = create_plots()
    kpis = compute_kpis()
    return render_template("performance.html", performance_plot=performance_plot, kpis=kpis)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_dict = {
        "Age": data["Age"],
        "Year": data["Year"],
        "Month": data["Month"],
        "Day": data["Day"],
        "Gender_Male": 1 if data["Gender"] == "Male" else 0
    }

    categories = ["Electronics", "Clothing", "Food", "Beauty", "Sports"]
    for cat in categories:
        col = f"Product Category_{cat}"
        input_dict[col] = 1 if data["Category"] == cat else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_df)[0]
    return jsonify({"prediction": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
