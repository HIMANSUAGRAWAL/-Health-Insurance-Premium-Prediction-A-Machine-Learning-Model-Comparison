import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\himan\Desktop\CUTM DOMAIN WORK\PREDICTION\Health Insurance\Health_insurance.csv")
print(data.head())

print(data.info())
print("-"*50)
print(data['region'].value_counts().sort_values())
print("-"*50)
print(data['children'].value_counts().sort_values())
print("-"*50)

clean_data = {'sex': {'male': 0, 'female': 1},
              'smoker': {'no': 0, 'yes': 1},
              'region': {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
              }
data_copy = data.copy()
data_copy.replace(clean_data, inplace=True)

print(data_copy.head())

corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='BuPu', annot=True, fmt=".2f", ax=ax)
plt.title("Dependencies of Medical Charges")


print(data['sex'].value_counts().sort_values())
print("-"*50)
print(data['smoker'].value_counts().sort_values())
print("-"*50)
print(data['region'].value_counts().sort_values())
print("-"*50)


plt.figure(figsize=(12, 9))
plt.title('Age vs Charge')
sns.barplot(x='age', y='charges', data=data_copy, palette='husl')

plt.figure(figsize=(10, 7))
plt.title('Region vs Charge')
sns.barplot(x='region', y='charges', data=data_copy, palette='Set3')

plt.figure(figsize=(7, 5))
sns.scatterplot(x='bmi', y='charges', hue='sex', data=data_copy, palette='Reds')
plt.title('BMI VS Charge')

plt.figure(figsize=(10, 7))
plt.title('Smoker vs Charge')
sns.barplot(x='smoker', y='charges', data=data_copy, palette='Blues', hue='sex')

plt.figure(figsize=(10, 7))
plt.title('Sex vs Charges')
sns.barplot(x='sex', y='charges', data=data_copy, palette='Set1')

print('Printing Skewness and Kurtosis for all columns')
print()
for col in list(data_copy.columns):
    print('{0} : Skewness {1:.3f} and  Kurtosis {2:.3f}'.format(col, data_copy[col].skew(), data_copy[col].kurt()))
print("-"*50)


plt.figure(figsize=(10, 7))
sns.distplot(data_copy['age'])
plt.title('Plot for Age')
plt.xlabel('Age')
plt.ylabel('Count')

plt.figure(figsize=(10, 7))
sns.distplot(data_copy['bmi'])
plt.title('Plot for BMI')
plt.xlabel('BMI')
plt.ylabel('Count')

plt.figure(figsize=(10, 7))
sns.distplot(data_copy['charges'])
plt.title('Plot for charges')
plt.xlabel('charges')
plt.ylabel('Count')


from sklearn.preprocessing import StandardScaler
data_pre = data_copy.copy()

tempBmi = data_pre.bmi
tempBmi = tempBmi.values.reshape(-1, 1)
data_pre['bmi'] = StandardScaler().fit_transform(tempBmi)

tempAge = data_pre.age
tempAge = tempAge.values.reshape(-1, 1)
data_pre['age'] = StandardScaler().fit_transform(tempAge)

tempCharges = data_pre.charges
tempCharges = tempCharges.values.reshape(-1, 1)
data_pre['charges'] = StandardScaler().fit_transform(tempCharges)

print(data_pre.head())

# Step 1: Prepare the dataset with all specified features
X_full = data_pre[['age', 'sex', 'smoker', 'region', 'children', 'bmi']].values  # All features
y_full = data_pre['charges'].values.reshape(-1, 1)  # Target variable

# Step 2: Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Print sizes of the training and testing sets
print('Size of X_train_full : ', X_train_full.shape)
print("-" * 50)
print('Size of y_train_full : ', y_train_full.shape)
print("-" * 50)
print('Size of X_test_full : ', X_test_full.shape)
print("-" * 50)
print('Size of y_test_full : ', y_test_full.shape)
print("-" * 50)

# Step 3: Train the Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor

# Define the Random Forest Regressor model
rf_full_model = RandomForestRegressor(random_state=42)

# Train the model
rf_full_model.fit(X_train_full, y_train_full.ravel())

# Step 4: Check feature importance
importances_full = rf_full_model.feature_importances_

# Step 5: Visualize feature importance
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': ['Age', 'Sex', 'Smoker', 'Region', 'Children', 'BMI'],
    'Importance': importances_full
}).sort_values(by='Importance', ascending=False)

# Plotting the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance from Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')

# Print the importance values
print(feature_importance_df)


# Assuming data_pre has been preprocessed and includes the necessary features

# Step 1: Select only the four most important features
X = data_pre[['children', 'age', 'smoker', 'bmi']].values  # Select only the four features
y = data_pre['charges'].values.reshape(-1, 1)  # Target variable remains the same

# Step 2: Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print sizes of the training and testing sets
print('Size of X_train : ', X_train.shape)
print("-" * 50)
print('Size of y_train : ', y_train.shape)
print("-" * 50)
print('Size of X_test : ', X_test.shape)
print("-" * 50)
print('Size of y_test : ', y_test.shape)
print("-" * 50)

# Linear Regression
print("\nLinear Regression Model - Performance")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import time

# Start the timer
start_time = time.time()
# Fit the linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
# End the timer
end_time = time.time()
# Print the elapsed time
print(f"Execution time: {end_time - start_time} seconds")

y_pred_linear_reg_train = linear_reg.predict(X_train)
r2_score_linear_reg_train = r2_score(y_train, y_pred_linear_reg_train)

y_pred_linear_reg_test = linear_reg.predict(X_test)
r2_score_linear_reg_test = r2_score(y_test, y_pred_linear_reg_test)

rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_reg_test)))

print('R2_score (train) : {0:.3f}'.format(r2_score_linear_reg_train))
print("-"*50)
print('R2_score (test) : {0:.3f}'.format(r2_score_linear_reg_test))
print("-"*50)
print('RMSE : {0:.3f}'.format(rmse_linear))
print("-"*50)


# Random Forest Regressor
print("\nRandom Forest Regressor Model - Performance")
from sklearn.ensemble import RandomForestRegressor

# Define the Random Forest Regressor model
rf = RandomForestRegressor(random_state=42)

# Train the Random Forest Regressor with the parameters
rf.fit(X_train, y_train.ravel())

# Performance evaluation on training data
y_pred_rf_train = rf.predict(X_train)
r2_score_rf_train = r2_score(y_train, y_pred_rf_train)

# Performance evaluation on test data
y_pred_rf_test = rf.predict(X_test)
r2_score_rf_test = r2_score(y_test, y_pred_rf_test)

# Calculate RMSE
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))

# Print performance metrics
print('R2 Score (train) : {0:.3f}'.format(r2_score_rf_train))
print("-" * 50)
print('R2 Score (test) : {0:.3f}'.format(r2_score_rf_test))
print("-" * 50)
print('RMSE : {0:.3f}'.format(rmse_rf))
print("-" * 50)

# Decision Tree Regressor
print("\nDecision Tree Regressor Model - Performance")
from sklearn.tree import DecisionTreeRegressor

# Define the Decision Tree Regressor model
dt = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree Regressor model
dt.fit(X_train, y_train)

# Performance evaluation on training data
y_pred_dt_train = dt.predict(X_train)
r2_score_dt_train = r2_score(y_train, y_pred_dt_train)

# Performance evaluation on test data
y_pred_dt_test = dt.predict(X_test)
r2_score_dt_test = r2_score(y_test, y_pred_dt_test)

# Calculate RMSE for Decision Tree
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt_test))

# Print performance metrics for Decision Tree
print('R2 Score (train) : {0:.3f}'.format(r2_score_dt_train))
print("-" * 50)
print('R2 Score (test) : {0:.3f}'.format(r2_score_dt_test))
print("-" * 50)
print('RMSE : {0:.3f}'.format(rmse_dt))
print("-" * 50)

models = [('Linear Regression', rmse_linear, r2_score_linear_reg_train, r2_score_linear_reg_test),
          ('Random Forest Regression', rmse_rf, r2_score_rf_train, r2_score_rf_test),
          ('Decision Tree Regression', rmse_dt, r2_score_dt_train, r2_score_dt_test)
         ]

predict = pd.DataFrame(data=models, columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)'])
print(predict)
print("-"*50)


#Actual vs Predicted Values
y_pred_rf_test = rf.predict(X_test)
plt.figure(figsize=(10,7))
plt.scatter(y_test, y_pred_rf_test, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges (Random Forest)')


# Display the initial data for reference
print(data_copy.head())
print("-" * 50)

# Select only the four most important features
X_ = data_copy[['children', 'age', 'smoker', 'bmi']].values  # Select only the four features
y_ = data_copy['charges'].values.reshape(-1, 1)  # Target variable remains the same

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=42)

print('Size of X_train_ : ', X_train_.shape)
print("-" * 50)
print('Size of y_train_ : ', y_train_.shape)
print("-" * 50)
print('Size of X_test_ : ', X_test_.shape)
print("-" * 50)
print('Size of y_test_ : ', y_test_.shape)
print("-" * 50)

# Define the Random Forest Regressor model with pre-set parameters
rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7, n_estimators=1200)

# Train the Random Forest Regressor on the selected features
rf_reg.fit(X_train_, y_train_.ravel())

# Predicting for training data
y_pred_rf_train_ = rf_reg.predict(X_train_)
r2_score_rf_train_ = r2_score(y_train_, y_pred_rf_train_)

# Predicting for test data
y_pred_rf_test_ = rf_reg.predict(X_test_)
r2_score_rf_test_ = r2_score(y_test_, y_pred_rf_test_)

# Results on training and testing set
print("\nResults in Training and Testing Set")
print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train_))
print("-" * 50)
print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test_))
print("-" * 50)

# Visualizing the manual feature importance (if needed)
# Note: Random Forest automatically calculates feature importance during training.
# If you want to visualize the importance of these four features:
importances = rf_reg.feature_importances_
feature_names = ['children', 'age', 'smoker', 'bmi']

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names, palette='Blues')
plt.title('Feature Importance from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')


import numpy as np
from sklearn.metrics import r2_score

# Function to convert scaled output back to original values (since the charges were scaled)
def inverse_transform_charge(scaled_value, original_mean, original_std):
    return scaled_value * original_std + original_mean

# Mean and standard deviation of charges before scaling (replace with actual values from your dataset)
mean_charges = 13270.422265141257  # Example mean of 'charges' from original dataset
std_charges = 12110.011236693994   # Example std deviation of 'charges' from original dataset

# Assuming the RandomForestRegressor model is already trained and stored in variable 'rf'
# rf is the trained RandomForestRegressor model

# User input for prediction based on most important features
print("Please enter the following details for insurance cost prediction:")

# Taking user input
children = int(input("Enter number of children (e.g., 2): "))
age = float(input("Enter age (e.g., 25): "))
smoker = int(input("Enter smoker status (0 for no, 1 for yes): "))
bmi = float(input("Enter BMI (e.g., 28.5): "))

# Prepare the input features as an array (order: children, age, smoker, bmi)
input_features = np.array([children, age, smoker, bmi]).reshape(1, 4)

# Make prediction using the trained model
predicted_scaled_cost = rf.predict(input_features)[0]

# Inverse transform the predicted scaled value to get the original cost in rupees
predicted_cost_in_rupees = inverse_transform_charge(predicted_scaled_cost, mean_charges, std_charges)

# Print the predicted insurance cost
print(f"\nThe predicted insurance cost is: ₹{predicted_cost_in_rupees:.2f}")
print("Thank you for choosing Secure Health. We prioritize your health and well-being.")


# Make predictions using the correct feature indices
y_pred = rf.predict(X_test_[:, [0, 1, 2, 3]])  # Using the correct order: children, age, smoker, bmi
# Calculate R² score on the test data
r2_score_test = r2_score(y_test_, y_pred)

# Print the accuracy (R² score)
print(f"Model accuracy (R² score on test data): {r2_score_test:.3f}")
