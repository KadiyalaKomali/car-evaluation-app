import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("car_evaluation.csv")

# Fix: Use 'outcome' instead of 'class'
X = df.drop("outcome", axis=1)
y = df["outcome"]

# Encode categorical features
encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    encoders[column] = le

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
for name, encoder in encoders.items():
    with open(f"le_{name}.pkl", "wb") as f:
        pickle.dump(encoder, f)

# Save target encoder
with open("le_target.pkl", "wb") as f:
    pickle.dump(le_target, f)

print("✅ Model and encoders saved.")
