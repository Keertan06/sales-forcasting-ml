import requests

url = "http://127.0.0.1:5000/predict"

sample_data = {
    "Age": 30,
    "Year": 2024,
    "Month": 1,
    "Day": 10,
    "Gender_Male": 1,
    "Product Category_Electronics": 1,
    "Product Category_Clothing": 0,
    "Product Category_Food": 0
}

response = requests.post(url, json=sample_data)

print("Status Code:", response.status_code)
print("Response Text:", response.text)
