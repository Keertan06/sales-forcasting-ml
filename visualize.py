import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ---------- 1. Monthly Sales Trend ----------
monthly_sales = df.groupby(df['Date'].dt.to_period("M"))['Total Amount'].sum()

plt.figure(figsize=(12,5))
monthly_sales.plot()
plt.title("Monthly Total Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/monthly_sales_trend.png")
plt.show()

# ---------- 2. Category-wise Revenue ----------
category_sales = df.groupby("Product Category")['Total Amount'].sum()

plt.figure(figsize=(8,5))
category_sales.plot(kind='bar')
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/category_revenue.png")
plt.show()

# ---------- 3. Gender-wise Spending ----------
gender_sales = df.groupby("Gender")['Total Amount'].sum()

plt.figure(figsize=(6,4))
gender_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Spending Distribution by Gender")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outputs/gender_spending.png")
plt.show()

# ---------- 4. Actual vs Predicted ----------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df_encoded = pd.get_dummies(df, columns=['Gender', 'Product Category'], drop_first=True)
df_encoded = df_encoded.drop(['Transaction ID', 'Customer ID', 'Date'], axis=1)

X = df_encoded.drop(['Total Amount', 'Quantity', 'Price per Unit'], axis=1)
y = df_encoded['Total Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()
