import pandas as pd
# Load the dataset
df = pd.read_csv("C:/Users/ISHITA/Desktop/Customer/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check the first few rows of the dataset
df.head()
# Get basic info about the dataset
df.info()

# View descriptive statistics
df.describe()

# Checking for missing values
df.isnull().sum()
# Fill missing values with the mode (or drop rows/columns)
#print(df.column)
# Fill missing values in 'TotalCharges' column with the mode
df['TotalCharges'].fillna(df['TotalCharges'].mode()[0], inplace=True)

# Example: One-hot encoding categorical columns
df = pd.get_dummies(df, columns=['column_name'], drop_first=True)
# Correlation matrix for numerical features
correlation_matrix = df.corr()

# Visualize correlation matrix (optional)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
# Get the coefficients of the logistic regression model
coefficients = model.coef_

# Create a DataFrame to view feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients[0]
})

# Sort the features by importance
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print(feature_importance)
