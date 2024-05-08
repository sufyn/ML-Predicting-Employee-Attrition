# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("IBM_HR_Employee_Attrition.csv")  # Replace with your filename

# Explore data
print(data.head())  # View first few rows
print(data.info())  # View data types and missing values
# print(data.corr())
# Analyze target variable
attrition_count = data["Attrition"].value_counts()
print(attrition_count)  # Count of employees leaving vs staying

# Explore categorical features
categorical_features = data.select_dtypes(include=["object"])
for col in categorical_features.columns:
  sns.countplot(x=col, data=data)  # Plot count for each category
  plt.show()



# Explore numerical features
# numerical_features = data.select_dtypes(include=["int64", "float64"])
# for col in numerical_features.columns:
#   sns.histplot(data[col], kde=True)  # Plot histogram with kernel density estimate
#   plt.show()

# Correlation analysis
# correlation_matrix = data.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.show()


# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Download dataset (replace with your download path)
# data_path = "IBM_HR_Employee_Attrition.csv"

# Read data
# df = pd.read_csv(data_path)
df = data
# Define features and target variable
features = [col for col in df.columns if col != "Attrition"]
target = "Attrition"

# Data Preprocessing

# Handle missing values (replace with your chosen imputation strategy)
# Example: fill numerical missing values with mean and categorical with mode
df["DistanceFromHome"].fillna(df["DistanceFromHome"].mean(), inplace=True)
for col in df.select_dtypes(include=['object']):
  df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
  df[col] = le.fit_transform(df[col])

# Scale numerical features (if necessary)
scaler = StandardScaler()
# df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)




from sklearn.feature_selection import SelectFromModel  # For feature selection
from imblearn.over_sampling import SMOTE  # Import for oversampling

# Encode categorical features
# Define numerical_features
numerical_features = [col for col in df.columns if col not in df.select_dtypes(include=['object'])]

# Scale numerical features (if necessary)
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Oversample the minority class (optional)
if df["Attrition"].value_counts().min() < df["Attrition"].value_counts().max():
  oversample = SMOTE(random_state=42)
  X_train, y_train = oversample.fit_resample(X_train, y_train)
  print(f"Class Distribution (After Oversampling):")
  print(pd.Series(y_train).value_counts())

# Feature Selection (using Random Forest as feature importance estimator)

# Train a Random Forest model
rf_estimator = RandomForestClassifier(random_state=42)
rf_estimator.fit(X_train, y_train)

# Select features based on importance scores
selector = SelectFromModel(rf_estimator, threshold=0.05)  # Select features with importance > 0.05
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]
print(f"Selected Features: {selected_features}")

# Reduced feature set for model training
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]

# Model Development and Evaluation

# Define models (consider adding more models if desired)
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Train and evaluate models
for model_name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print(f"** Model: {model_name} **")
  print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
  print(f"Precision: {precision_score(y_test, y_pred,zero_division='warn')}")
  print(f"Recall: {recall_score(y_test, y_pred)}")
  print(f"F1-Score: {f1_score(y_test, y_pred)}")
  print("-" * 30)

# Model Optimization (using GridSearchCV for Random Forest as an example)

# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_micro')
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("** Optimized Random Forest Model **")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best F1-Score: {grid_search.best_score_}")

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("** Best Model Evaluation **")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred,zero_division='warn')}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print("-" * 30)

# Classification Report
print("** Classification Report **")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("** Confusion Matrix **")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
# Extract feature importance from the best model
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
print("** Feature Importance **")
print(feature_importance_df)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance")
plt.show()

# Save the best model
import joblib

joblib.dump(best_model, "best_model.pkl")
print("Model saved successfully!")


