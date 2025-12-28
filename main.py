import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib

# 1. Load the dataset:
df = pd.read_csv('/content/parkinsons.csv')

print(df.head())

# 2. Select features:

# input_features = ['PPE', 'DFA']
input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']

# 0 = Healthy Control
# 1 = Parkinson's Disease (PD)
output_feature = 'status'

X = df[input_features]
y = df[output_feature]

print("Input Features (X):")
print(X.head())
print("\nOutput Variable (y):")
print(y.head())

# 3. Scale the data:
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data:
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=29
)

# 5. Choose a model:
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=29)

model.fit(X_train, y_train)

# 6. Test the accuracy:
# Generate predictions using the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

# Verification check
if accuracy >= 0.8:
    print("Requirement Met: The accuracy is above 0.8.")
else:
    print("Requirement Not Met: The accuracy is below 0.8. You may need to tune the model or check the data split.")

# 7. Save and upload the model:
joblib.dump(model, 'my_model.joblib')

