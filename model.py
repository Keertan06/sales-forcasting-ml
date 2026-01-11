import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("data/retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encode categorical columns
df = pd.get_dummies(df, columns=['Gender', 'Product Category'], drop_first=True)

# Drop ID and Date columns (not useful for prediction)
df = df.drop(['Transaction ID', 'Customer ID', 'Date'], axis=1)

# Features & Target
X = df.drop(['Total Amount', 'Quantity', 'Price per Unit'], axis=1)
y = df['Total Amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Model MAE:", mae)

# Save model and feature names
joblib.dump(model, "sales_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("Model and feature schema saved.")

