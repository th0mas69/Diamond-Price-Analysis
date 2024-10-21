import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/tluke/diamond_price_analysis_sample.csv')  

# Step 2: Data Preprocessing
# Drop unnecessary columns (if any) and handle missing values
df = df.dropna()

# Encode categorical features ('cut', 'color', 'clarity')
le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])
df['color'] = le.fit_transform(df['color'])
df['clarity'] = le.fit_transform(df['clarity'])

# Scaling the numeric features
scaler = StandardScaler()
df[['carat', 'depth', 'table', 'x', 'y', 'z']] = scaler.fit_transform(df[['carat', 'depth', 'table', 'x', 'y', 'z']])

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10,6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution of Diamond Prices")
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(df[['price', 'carat', 'depth', 'table', 'x', 'y', 'z']])
plt.show()

# Step 4: Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 5: Statistical Analysis
print(df.describe())

# Step 6: Modeling and Prediction
# Split the data into training and testing sets
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Gradient Boosting Regressor Model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Step 7: Price Prediction Using Machine Learning
# Calculate the performance of each model
print("Linear Regression - Mean Squared Error: ", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression - R2 Score: ", r2_score(y_test, y_pred_lr))

print("Random Forest - Mean Squared Error: ", mean_squared_error(y_test, y_pred_rf))
print("Random Forest - R2 Score: ", r2_score(y_test, y_pred_rf))

print("Gradient Boosting - Mean Squared Error: ", mean_squared_error(y_test, y_pred_gb))
print("Gradient Boosting - R2 Score: ", r2_score(y_test, y_pred_gb))

# Step 8: Interpretation of Results
# Feature Importance for Random Forest
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], color='b', align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.show()

# Final Model Selection (Based on Evaluation)
best_model = rf_model if r2_score(y_test, y_pred_rf) > r2_score(y_test, y_pred_gb) else gb_model
print("Best model is chosen based on R2 score and Mean Squared Error.")
