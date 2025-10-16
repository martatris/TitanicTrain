# Titanic Data Analysis Project

## Overview
This project demonstrates a complete **data analysis and machine learning workflow** using the **Titanic dataset**. The objective is to explore the data, perform feature engineering, and build a predictive model to classify whether a passenger survived or not. The project is implemented in **Python** using popular libraries like **pandas, seaborn, matplotlib, and scikit-learn**.

---

## Dataset
The dataset used in this project is from **Kaggle: Titanic – Machine Learning from Disaster** ([link](https://www.kaggle.com/c/titanic/data)).

- **train.csv** – Used to train the machine learning model  
- **test.csv** – Used to make predictions for submission  
- Contains columns like `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

---

## Project Steps

1. **Import Libraries**  
   Load necessary Python packages for data manipulation, visualization, and modeling.

2. **Load Dataset**  
   Load the CSV files (`train.csv` and optionally `test.csv`) into pandas DataFrames.

3. **Data Exploration (EDA)**  
   - Examine dataset structure and summary statistics  
   - Visualize distributions and relationships (e.g., survival count, survival by gender)

4. **Data Cleaning**  
   - Handle missing values (Age, Embarked, Fare)  
   - Drop irrelevant columns (`Cabin`, `Ticket`, `Name`)  

5. **Feature Engineering**  
   - Convert categorical variables to numeric using `LabelEncoder`  
   - Create new features like `FamilySize`  

6. **Train/Test Split**  
   Split the dataset into training and testing sets for model evaluation.

7. **Model Training**  
   Train a **Random Forest Classifier** to predict survival.

8. **Model Evaluation**  
   - Evaluate accuracy on test data  
   - Plot confusion matrix  
   - Generate classification report  
   - Identify important features

9. **Prediction on New Data**  
   Apply preprocessing to `test.csv` and generate predictions for submission.

---

## Libraries Used
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

## How to Run
1. Download `train.csv` and `test.csv` from Kaggle.  
2. Place them in the project directory.  
3. Run `titanic_analysis.py` or open the Jupyter Notebook version.  
4. Check outputs:  
   - Visualizations (plots)  
   - Model evaluation metrics  
   - `titanic_predictions.csv` file for test data predictions  

---

## Results
- Achieved good classification accuracy using Random Forest  
- Important features influencing survival: `Sex`, `Fare`, `Pclass`, `FamilySize`, `Age`  

---

## Notes
- This project is suitable for beginners to intermediate learners who want to practice **data cleaning, EDA, and predictive modeling** in Python.  
- Can be extended with other algorithms (Logistic Regression, XGBoost) or additional feature engineering.
