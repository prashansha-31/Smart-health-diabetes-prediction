import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("🚀 Training started")

# Load dataset
data = pd.read_csv("datasets/diabetes.csv")

# SELECT ONLY 3 FEATURES
X = data[["Glucose", "BMI", "Age"]]
y = data["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save new model
pickle.dump(model, open("models/diabetes_3feature_model.pkl", "wb"))

print("✅ Model trained with 3 features and saved")
