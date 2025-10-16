# Basic packages
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('............................................')
print('Load the Dataset')
# Load Titanic dataset
titanic = pd.read_csv("titanic/train.csv")

# Quick overview
print(titanic.shape)
print(titanic.info())
titanic.head()

print('............................................')
print('Basic Exploration')
# Summary statistics
print(titanic.describe())

# Check missing values
print(titanic.isnull().sum())

# Survival count
sns.countplot(x='Survived', data=titanic)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.show()

# Survival by gender
sns.countplot(x='Survived', hue='Sex', data=titanic)
plt.title("Survival by Gender")
plt.show()

print('............................................')
print('Data Cleaning')
# Fill missing Age values with median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Fill missing Embarked values with most common value
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
titanic.drop(columns=['Cabin'], inplace=True)

# Drop Ticket and Name (not useful for modeling)
titanic.drop(columns=['Ticket', 'Name'], inplace=True)

print('............................................')
print('Feature Engineering')
# Convert categorical variables to numeric
le = LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex'])
titanic['Embarked'] = le.fit_transform(titanic['Embarked'])

# Create new feature: FamilySize
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

# Drop PassengerId for training
X = titanic.drop(columns=['Survived', 'PassengerId'])
y = titanic['Survived']

X.head()

print('............................................')
print('Split Data into Train and Test Set')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print('............................................')
print('Train a Model using Random Forest')
###### Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('............................................')
print('Evaluate the Model')
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))


print('............................................')
print('Feature Importance')
# Check which features are most important
importances = rf.feature_importances_
feature_names = X.columns

feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp = feature_imp.sort_values(by='Importance', ascending=False)

# Plot feature importance
sns.barplot(x='Importance', y='Feature', data=feature_imp)
plt.title("Feature Importance")
plt.show()

print('............................................')
print('Predict on New Data')
test_data = pd.read_csv("titanic/test.csv")

# Perform the same preprocessing steps
test_data['Age'].fillna(titanic['Age'].median(), inplace=True)
test_data['Fare'].fillna(titanic['Fare'].median(), inplace=True)
test_data['Sex'] = le.fit_transform(test_data['Sex'])
test_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'] = le.fit_transform(test_data['Embarked'])
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Drop unnecessary columns
test_data.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')

# Predict
test_pred = rf.predict(test_data.drop(columns=['PassengerId']))
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_pred
})
submission.to_csv("titanic_predictions.csv", index=False)
print("Submission file saved!")












