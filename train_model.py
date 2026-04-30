import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/car_data.csv")
df.columns = [col.lower() for col in df.columns]

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Extract brand
df['brand'] = df['name'].apply(lambda x: x.split()[0].lower())

# Drop name
df.drop(['name'], axis=1, inplace=True)

# Convert owner to string
df['owner'] = df['owner'].astype(str)

# Create car age
df['car_age'] = 2026 - df['year']
df.drop(['year'], axis=1, inplace=True)

# -----------------------------
# 📊 EDA VISUALIZATION
# -----------------------------

plt.figure()
sns.histplot(df['selling_price'], kde=True)
plt.title("Selling Price Distribution")
plt.savefig("price_distribution.png")

plt.figure()
sns.scatterplot(x=df['car_age'], y=df['selling_price'])
plt.title("Car Age vs Price")
plt.savefig("age_vs_price.png")

plt.figure(figsize=(10,5))
df['brand'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Car Brands")
plt.savefig("brand_distribution.png")

# -----------------------------
# ENCODING
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# MODEL
# -----------------------------
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
pred = model.predict(X_test)

plt.figure()
sns.scatterplot(x=y_test, y=pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.savefig("prediction_plot.png")

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("✅ Model + Graphs saved successfully!")